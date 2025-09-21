import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import math
import pydeck as pdk
import os
import random

# Force use of Plotly maps and pydeck for better performance
FOLIUM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Ethiopia Electrification Decision Support Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #d0d0d0;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .highlight {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .step-complete {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .step-pending {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .pre-step-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare both CSV files"""
    # Load full dataset
    dre_atlas_df = pd.read_csv("dre_atlas.csv")
    
    # Create unelectrified dataset if it doesn't exist
    if not os.path.exists("unelectrified.csv"):
        # Filter for unelectrified settlements (has_nightlight == False)
        unelectrified_df = dre_atlas_df[dre_atlas_df['has_nightlight'] == False].copy()
        unelectrified_df.to_csv("unelectrified.csv", index=False)
        st.success("✅ Created unelectrified.csv with filtered settlements")
    else:
        unelectrified_df = pd.read_csv("unelectrified.csv")
    
    return dre_atlas_df, unelectrified_df

@st.cache_data
def load_sample_data():
    """Generate sample settlement data for demonstration"""
    np.random.seed(42)
    n_settlements = 1000
    
    # Generate synthetic data based on DRE Atlas structure
    data = {
        'geohash': [f"eth_{i:04d}" for i in range(n_settlements)],
        'lat': np.random.uniform(3.0, 15.0, n_settlements),
        'lon': np.random.uniform(33.0, 48.0, n_settlements),
        'village_name': [f"Village_{i}" for i in range(n_settlements)],
        'admin_cgaz_1': np.random.choice(['Oromia', 'Amhara', 'SNNP', 'Tigray', 'Somali'], n_settlements),
        'admin_cgaz_2': [f"District_{i%50}" for i in range(n_settlements)],
        'population': np.random.lognormal(5, 1, n_settlements).astype(int),
        'num_connections': lambda pop: (pop * np.random.uniform(0.15, 0.25, len(pop))).astype(int),
        'demand': np.random.uniform(50, 500, n_settlements),
        'demand_connection': np.random.uniform(0.3, 2.5, n_settlements),
        'distance_to_existing_transmission_lines': np.random.exponential(15, n_settlements),
        'main_road_access': np.random.choice([0, 1], n_settlements, p=[0.4, 0.6]),
        'dist_main_road_km': np.random.exponential(5, n_settlements),
        'pv_value': np.random.uniform(1600, 2100, n_settlements),
        'mean_rwi': np.random.normal(0, 1, n_settlements),
        'security_risk': np.random.choice(['low', 'medium', 'high'], n_settlements, p=[0.6, 0.3, 0.1]),
        'has_education_facility': np.random.choice([0, 1], n_settlements, p=[0.7, 0.3]),
        'has_health_facility': np.random.choice([0, 1], n_settlements, p=[0.8, 0.2]),
        'has_nightlight': np.random.choice([0, 1], n_settlements, p=[0.7, 0.3]),  # Add nightlight status
        'electrified': np.zeros(n_settlements, dtype=bool),
        'electrification_year': np.zeros(n_settlements, dtype=int),
        'technology_chosen': [''] * n_settlements,
        'cost_per_connection': np.zeros(n_settlements),
        'carbon_avoided_annual': np.zeros(n_settlements),
        'total_cost': np.zeros(n_settlements)
    }
    
    # Calculate num_connections based on population
    data['num_connections'] = (data['population'] * np.random.uniform(0.15, 0.25, n_settlements)).astype(int)
    
    df = pd.DataFrame(data)
    return df

class ElectrificationModel:
    def __init__(self, settlements_df):
        self.settlements = settlements_df.copy()
        self.yearly_results = []
        self.full_cost_results = None
        
    def calculate_technology_costs(self, settlement_row, params):
        """Calculate costs for each technology option for a settlement"""
        costs = {}
        reliability_scores = {}
        
        # Grid Extension Cost
        grid_distance = settlement_row['distance_to_existing_transmission_lines']
        road_penalty = params['road_penalty'] if not settlement_row['main_road_access'] else 1.0
        
        grid_cost = (
            params['grid_base_cost'] + 
            (grid_distance * params['grid_per_km_cost']) + 
            (settlement_row['num_connections'] * params['grid_per_connection_cost'])
        ) * road_penalty
        
        costs['grid'] = grid_cost
        reliability_scores['grid'] = 0.95  # Grid is most reliable
        
        # Mini-grid Cost
        minigrid_cost = settlement_row['num_connections'] * params['minigrid_per_connection_cost'] * road_penalty
        costs['minigrid'] = minigrid_cost
        reliability_scores['minigrid'] = 0.85  # Mini-grid is moderately reliable
        
        # Solar Home System Cost
        shs_cost = settlement_row['num_connections'] * params['shs_per_connection_cost'] * road_penalty
        costs['shs'] = shs_cost
        reliability_scores['shs'] = 0.60  # SHS is least reliable (individual systems)
        
        return costs, reliability_scores
    
    def select_best_technology(self, settlement_row, params):
        """Select best technology based on settlement characteristics and thresholds"""
        costs, reliability_scores = self.calculate_technology_costs(settlement_row, params)
        
        # Get settlement characteristics
        grid_distance = settlement_row['distance_to_existing_transmission_lines']
        population = settlement_row['population']
        connections = settlement_row['num_connections']
        has_road = settlement_row['main_road_access']
        demand_per_connection = settlement_row['demand_connection']
        
        # Define technology suitability thresholds
        # These create natural boundaries for technology selection
        
        # GRID EXTENSION RULES
        # Grid is best for: close proximity OR large settlements
        grid_distance_threshold = 15  # km - within this distance, grid is usually best
        grid_population_threshold = 2000  # Above this population, grid becomes attractive
        grid_connection_threshold = 300  # Above this, economies of scale favor grid
        
        # MINI-GRID RULES  
        # Mini-grid is best for: medium distance, clustered settlements
        minigrid_distance_min = 10  # km - beyond this distance from grid
        minigrid_distance_max = 50  # km - within this distance (not too remote)
        minigrid_population_min = 200  # Minimum population for mini-grid viability
        minigrid_population_max = 2000  # Maximum before grid becomes better
        minigrid_demand_threshold = 0.8  # Higher demand per connection favors mini-grid
        
        # SHS RULES
        # SHS is best for: remote, small, scattered settlements
        shs_distance_threshold = 30  # km - beyond this, SHS becomes attractive
        shs_population_threshold = 500  # Below this, SHS is often best
        shs_connection_threshold = 50  # Very small settlements favor SHS
        
        # Calculate technology scores with suitability bonuses
        tech_scores = {}
        
        # GRID SCORE
        grid_score = 0
        if grid_distance <= grid_distance_threshold:
            grid_score += 2.0  # Strong bonus for proximity
        if population >= grid_population_threshold:
            grid_score += 1.5  # Bonus for large population
        if connections >= grid_connection_threshold:
            grid_score += 1.0  # Bonus for many connections
        if has_road:
            grid_score += 0.5  # Road access helps grid extension
        # Add cost factor
        grid_score += (1.0 / (1 + costs['grid'] / 100000))  # Normalized cost factor
        # Add reliability factor
        grid_score += reliability_scores['grid'] * params.get('reliability_weight', 0.3)
        
        # MINI-GRID SCORE
        minigrid_score = 0
        if minigrid_distance_min <= grid_distance <= minigrid_distance_max:
            minigrid_score += 2.0  # Strong bonus for medium distance
        if minigrid_population_min <= population <= minigrid_population_max:
            minigrid_score += 1.5  # Bonus for medium population
        if demand_per_connection >= minigrid_demand_threshold:
            minigrid_score += 1.0  # Bonus for productive use potential
        if not has_road and grid_distance > 20:
            minigrid_score += 0.8  # Mini-grids can work without roads better than grid
        # Add cost factor
        minigrid_score += (1.0 / (1 + costs['minigrid'] / 100000))
        # Add reliability factor
        minigrid_score += reliability_scores['minigrid'] * params.get('reliability_weight', 0.3)
        
        # SHS SCORE
        shs_score = 0
        if grid_distance >= shs_distance_threshold:
            shs_score += 2.0  # Strong bonus for remote locations
        if population <= shs_population_threshold:
            shs_score += 1.5  # Bonus for small settlements
        if connections <= shs_connection_threshold:
            shs_score += 1.5  # Bonus for very small settlements
        if not has_road:
            shs_score += 0.5  # SHS doesn't need road infrastructure
        # Add cost factor (SHS is often cheapest)
        shs_score += (1.0 / (1 + costs['shs'] / 100000))
        # Add reliability factor (lower for SHS)
        shs_score += reliability_scores['shs'] * params.get('reliability_weight', 0.3)
        
        # Apply a diversity factor to encourage technology mix
        # This slightly randomizes close decisions to create more realistic mix
        # Use settlement's geohash as seed for reproducibility
        random.seed(hash(settlement_row['geohash']) % 2**32)
        diversity_factor = params.get('technology_diversity', 0.1)
        if diversity_factor > 0:
            grid_score += random.uniform(-diversity_factor, diversity_factor)
            minigrid_score += random.uniform(-diversity_factor, diversity_factor)
            shs_score += random.uniform(-diversity_factor, diversity_factor)
        
        # Store scores for analysis
        tech_scores = {
            'grid': grid_score,
            'minigrid': minigrid_score,
            'shs': shs_score
        }
        
        # Select technology with highest score
        best_tech = max(tech_scores.keys(), key=lambda x: tech_scores[x])
        best_cost = costs[best_tech]
        
        # Special case overrides for extreme situations
        # Force grid for very close and large settlements
        if grid_distance < 5 and population > 3000:
            best_tech = 'grid'
            best_cost = costs['grid']
        # Force SHS for very remote and tiny settlements  
        elif grid_distance > 75 and connections < 20:
            best_tech = 'shs'
            best_cost = costs['shs']
        # Force mini-grid for ideal mini-grid conditions
        elif (15 <= grid_distance <= 40 and 
              500 <= population <= 1500 and 
              demand_per_connection > 1.0):
            best_tech = 'minigrid'
            best_cost = costs['minigrid']
        
        return best_tech, best_cost, reliability_scores[best_tech]
    
    def calculate_carbon_finance(self, settlement_row, technology, params):
        """Calculate carbon finance revenue for a settlement"""
        # Baseline emissions (kerosene/diesel per household per year)
        baseline_emissions = params['baseline_emissions_per_hh'] * settlement_row['num_connections']
        
        # Technology emissions
        if technology == 'grid':
            tech_emissions = baseline_emissions * params['grid_emission_factor']
        elif technology == 'minigrid':
            tech_emissions = 0  # Assume renewable minigrid
        else:  # shs
            tech_emissions = 0  # Solar is clean
            
        avoided_emissions = max(0, baseline_emissions - tech_emissions)
        carbon_revenue = avoided_emissions * params['carbon_price']
        
        return avoided_emissions, carbon_revenue
    
    def calculate_priority_score(self, settlement_row, cost_per_connection, params):
        """Calculate priority score for settlement electrification with weighted factors"""
        scores = {}
        
        # 1. Population Scale Score (serve more people)
        max_pop = 10000  # Normalization factor
        scores['population'] = min(settlement_row['population'] / max_pop, 1.0)
        
        # 2. Poverty Focus Score (prioritize poor communities)
        # Lower wealth index = higher priority (inverse relationship)
        rwi = settlement_row['mean_rwi']
        scores['poverty'] = max(0, min(1, 1.0 - (rwi + 2) / 4))  # Normalize RWI from [-2,2] to [1,0]
        
        # 3. Cost Efficiency Score (lower cost per connection is better)
        max_cost = 5000  # Maximum expected cost per connection
        scores['cost_efficiency'] = max(0, 1.0 - (cost_per_connection / max_cost))
        
        # 4. Implementation Ease Score (road access and proximity to grid)
        road_score = 1.0 if settlement_row['main_road_access'] else 0.3
        grid_proximity = max(0, 1.0 - (settlement_row['distance_to_existing_transmission_lines'] / 100))
        scores['implementation_ease'] = (road_score * 0.6 + grid_proximity * 0.4)
        
        # 5. Social Infrastructure Score (schools and health facilities)
        facility_score = 0.0
        if settlement_row['has_education_facility']:
            facility_score += 0.5
        if settlement_row['has_health_facility']:
            facility_score += 0.5
        scores['social_infrastructure'] = facility_score
        
        # 6. Productive Use Potential Score (agricultural and economic activity)
        # Using demand per connection as proxy for productive use
        demand_score = min(settlement_row['demand_connection'] / 2.0, 1.0)  # Normalize by 2 kWh/day
        scores['productive_use'] = demand_score
        
        # 7. Security Score (avoid high-risk areas)
        security_map = {'low': 1.0, 'medium': 0.5, 'high': 0.1}
        scores['security'] = security_map.get(settlement_row['security_risk'], 0.5)
        
        # Apply weights from parameters
        weights = {
            'population': params.get('population_weight', 0.5),
            'poverty': params.get('poverty_weight', 0.5),
            'cost_efficiency': params.get('cost_efficiency_weight', 0.6),
            'implementation_ease': params.get('implementation_ease_weight', 0.4),
            'social_infrastructure': params.get('social_infrastructure_weight', 0.7),
            'productive_use': params.get('productive_use_weight', 0.4),
            'security': params.get('security_weight', 0.3)
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted priority score
        priority_score = sum(scores[factor] * weights[factor] for factor in scores.keys())
        
        # Store individual scores for debugging/analysis
        settlement_row['priority_scores'] = scores
        settlement_row['priority_weights'] = weights
        
        return priority_score
    
    def calculate_full_electrification_cost(self, params):
        """Calculate total cost to electrify ALL settlements (no budget constraint)"""
        settlements = self.settlements.copy()
        total_cost = 0
        total_connections = 0
        technology_breakdown = {'grid': 0, 'minigrid': 0, 'shs': 0}
        cost_breakdown = {'grid': 0, 'minigrid': 0, 'shs': 0}
        
        settlement_details = []
        
        for idx, settlement in settlements.iterrows():
            # Select best technology considering cost and reliability
            best_tech, best_cost, reliability = self.select_best_technology(settlement, params)
            
            # Calculate carbon finance
            co2_avoided, carbon_revenue = self.calculate_carbon_finance(settlement, best_tech, params)
            
            # Accumulate totals
            total_cost += best_cost
            total_connections += settlement['num_connections']
            technology_breakdown[best_tech] += settlement['num_connections']
            cost_breakdown[best_tech] += best_cost
            
            settlement_details.append({
                'geohash': settlement['geohash'],
                'village_name': settlement['village_name'],
                'technology': best_tech,
                'cost': best_cost,
                'connections': settlement['num_connections'],
                'co2_avoided': co2_avoided,
                'carbon_revenue': carbon_revenue,
                'cost_per_connection': best_cost / max(settlement['num_connections'], 1),
                'reliability_score': reliability
            })
        
        self.full_cost_results = {
            'total_cost': total_cost,
            'total_connections': total_connections,
            'technology_breakdown': technology_breakdown,
            'cost_breakdown': cost_breakdown,
            'settlement_details': settlement_details,
            'average_cost_per_connection': total_cost / max(total_connections, 1)
        }
        
        return self.full_cost_results
    
    def simulate_electrification(self, params, annual_budgets, start_year=2025):
        """Run the electrification simulation with flexible annual budgets"""
        self.settlements['electrified'] = False
        self.settlements['electrification_year'] = 0
        self.settlements['technology_chosen'] = ''
        self.settlements['cost_per_connection'] = 0
        self.settlements['carbon_avoided_annual'] = 0
        
        yearly_results = []
        years = sorted(annual_budgets.keys())
        
        for year in years:
            year_budget = annual_budgets[year]
            settlements_electrified = 0
            total_spent = 0
            total_carbon_revenue = 0
            total_co2_avoided = 0
            
            # Get unelectrified settlements
            unelectrified = self.settlements[~self.settlements['electrified']].copy()
            
            if len(unelectrified) == 0:
                break
            
            # Calculate costs and priorities for each settlement
            settlement_options = []
            
            for idx, settlement in unelectrified.iterrows():
                # Select best technology considering cost and reliability
                best_tech, best_cost, reliability = self.select_best_technology(settlement, params)
                cost_per_connection = best_cost / max(settlement['num_connections'], 1)
                
                # Calculate carbon finance
                co2_avoided, carbon_revenue = self.calculate_carbon_finance(settlement, best_tech, params)
                
                # Calculate priority score with all weighted factors
                priority_score = self.calculate_priority_score(settlement, cost_per_connection, params)
                
                settlement_options.append({
                    'index': idx,
                    'technology': best_tech,
                    'total_cost': best_cost,
                    'cost_per_connection': cost_per_connection,
                    'co2_avoided': co2_avoided,
                    'carbon_revenue': carbon_revenue,
                    'priority_score': priority_score,
                    'population': settlement['population'],
                    'num_connections': settlement['num_connections'],
                    'reliability': reliability,
                    'village_name': settlement['village_name']
                })
            
            # Sort by priority score (descending)
            settlement_options.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Electrify settlements within budget
            for option in settlement_options:
                # Add carbon revenue to effective budget
                effective_budget = year_budget + total_carbon_revenue
                
                if total_spent + option['total_cost'] <= effective_budget:
                    # Electrify this settlement
                    idx = option['index']
                    self.settlements.loc[idx, 'electrified'] = True
                    self.settlements.loc[idx, 'electrification_year'] = year
                    self.settlements.loc[idx, 'technology_chosen'] = option['technology']
                    self.settlements.loc[idx, 'cost_per_connection'] = option['cost_per_connection']
                    self.settlements.loc[idx, 'carbon_avoided_annual'] = option['co2_avoided']
                    self.settlements.loc[idx, 'total_cost'] = option['total_cost']
                    self.settlements.loc[idx, 'priority_score'] = option['priority_score']
                    
                    total_spent += option['total_cost']
                    total_carbon_revenue += option['carbon_revenue']
                    total_co2_avoided += option['co2_avoided']
                    settlements_electrified += 1
                else:
                    break
            
            # Store yearly results
            yearly_results.append({
                'year': year,
                'budget': year_budget,
                'settlements_electrified': settlements_electrified,
                'total_spent': total_spent,
                'carbon_revenue': total_carbon_revenue,
                'co2_avoided': total_co2_avoided,
                'cumulative_electrified': self.settlements['electrified'].sum(),
                'electrification_rate': self.settlements['electrified'].sum() / len(self.settlements) * 100
            })
        
        self.yearly_results = yearly_results
        return yearly_results

def create_parameter_sidebar():
    """Create sidebar with parameter controls"""
    st.sidebar.header("🔧 Model Parameters")
    
    # Technology Costs
    st.sidebar.subheader("⚡ Technology Costs")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        grid_base_cost = st.sidebar.number_input("Grid Base Cost ($)", value=50000, step=5000)
        grid_per_km_cost = st.sidebar.number_input("Grid Per km ($)", value=8000, step=500)
        grid_per_connection_cost = st.sidebar.number_input("Grid Per Connection ($)", value=500, step=50)
    
    with col2:
        minigrid_per_connection_cost = st.sidebar.number_input("Mini-grid Per Connection ($)", value=1200, step=100)
        shs_per_connection_cost = st.sidebar.number_input("SHS Per Connection ($)", value=300, step=25)
        road_penalty = st.sidebar.slider("Road Access Penalty", 1.0, 3.0, 1.5, 0.1)
    
    # Carbon Finance
    st.sidebar.subheader("🌱 Carbon Finance")
    carbon_price = st.sidebar.number_input("Carbon Price ($/ton CO₂)", value=15, step=1)
    baseline_emissions_per_hh = st.sidebar.number_input(
        "Baseline Emissions per HH (tons CO₂/year)", 
        value=1.2, step=0.1
    )
    
    # Grid Mix
    st.sidebar.subheader("🔌 Grid Mix")
    grid_emission_factor = st.sidebar.slider(
        "Grid Emission Factor (0=clean, 1=dirty)", 
        0.0, 1.0, 0.6, 0.1
    )
    
    # Technology Selection
    st.sidebar.subheader("🔧 Technology Selection")
    technology_diversity = st.sidebar.slider(
        "Technology Diversity Factor",
        0.0, 0.5, 0.1, 0.05,
        help="Higher values create more diverse technology mix (0=strict rules, 0.5=more random)"
    )
    
    # Prioritization Weights
    st.sidebar.subheader("🎯 Prioritization Strategy")
    st.sidebar.markdown("*Adjust weights to balance competing objectives:*")
    
    # Social Equity vs Scale
    population_weight = st.sidebar.slider(
        "Population Scale", 
        0.0, 1.0, 0.5, 0.05,
        help="Higher = prioritize larger settlements (serve more people)"
    )
    
    poverty_weight = st.sidebar.slider(
        "Poverty Focus", 
        0.0, 1.0, 0.5, 0.05,
        help="Higher = prioritize poor communities (lower wealth index)"
    )
    
    # Efficiency vs Speed
    cost_efficiency_weight = st.sidebar.slider(
        "Cost Efficiency", 
        0.0, 1.0, 0.6, 0.05,
        help="Higher = prioritize lower cost per connection"
    )
    
    implementation_ease_weight = st.sidebar.slider(
        "Implementation Speed", 
        0.0, 1.0, 0.4, 0.05,
        help="Higher = prioritize settlements with road access (easier to build)"
    )
    
    # Service Provision
    social_infrastructure_weight = st.sidebar.slider(
        "Social Infrastructure", 
        0.0, 1.0, 0.7, 0.05,
        help="Higher = prioritize settlements with schools/health facilities"
    )
    
    # Technology Reliability
    reliability_weight = st.sidebar.slider(
        "Technology Reliability", 
        0.0, 1.0, 0.3, 0.05,
        help="Higher = prefer grid/minigrid over SHS for reliability"
    )
    
    # Economic Development
    productive_use_weight = st.sidebar.slider(
        "Productive Use Potential", 
        0.0, 1.0, 0.4, 0.05,
        help="Higher = prioritize settlements with agricultural/commercial potential"
    )
    
    # Security & Risk
    security_weight = st.sidebar.slider(
        "Security Priority", 
        0.0, 1.0, 0.3, 0.05,
        help="Higher = prioritize secure areas (avoid high-risk zones)"
    )
    
    return {
        'grid_base_cost': grid_base_cost,
        'grid_per_km_cost': grid_per_km_cost,
        'grid_per_connection_cost': grid_per_connection_cost,
        'minigrid_per_connection_cost': minigrid_per_connection_cost,
        'shs_per_connection_cost': shs_per_connection_cost,
        'road_penalty': road_penalty,
        'carbon_price': carbon_price,
        'baseline_emissions_per_hh': baseline_emissions_per_hh,
        'grid_emission_factor': grid_emission_factor,
        'technology_diversity': technology_diversity,
        # Prioritization weights
        'population_weight': population_weight,
        'poverty_weight': poverty_weight,
        'cost_efficiency_weight': cost_efficiency_weight,
        'implementation_ease_weight': implementation_ease_weight,
        'social_infrastructure_weight': social_infrastructure_weight,
        'reliability_weight': reliability_weight,
        'productive_use_weight': productive_use_weight,
        'security_weight': security_weight
    }

def create_prestep_analysis(dre_atlas_df):
    """Create Pre-Step analysis showing current electrification status"""
    st.header("🔌 Pre-Step: Current Electrification Status")
    
    # Basic statistics
    total_settlements = len(dre_atlas_df)
    electrified = dre_atlas_df[dre_atlas_df['has_nightlight'] == True]
    unelectrified = dre_atlas_df[dre_atlas_df['has_nightlight'] == False]
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Settlements",
            f"{total_settlements:,}",
            help="Total number of settlements in Ethiopia"
        )
    
    with col2:
        electrified_count = len(electrified)
        electrification_rate = (electrified_count / total_settlements) * 100
        st.metric(
            "Electrified Settlements",
            f"{electrified_count:,}",
            f"{electrification_rate:.1f}%",
            help="Settlements with nightlight detected"
        )
    
    with col3:
        unelectrified_count = len(unelectrified)
        unelectrified_rate = (unelectrified_count / total_settlements) * 100
        st.metric(
            "Unelectrified Settlements",
            f"{unelectrified_count:,}",
            f"{unelectrified_rate:.1f}%",
            delta_color="inverse",
            help="Settlements without nightlight"
        )
    
    with col4:
        # Population statistics
        total_pop = dre_atlas_df['population'].sum()
        unelectrified_pop = unelectrified['population'].sum()
        unelectrified_pop_pct = (unelectrified_pop / total_pop) * 100
        st.metric(
            "Unelectrified Population",
            f"{unelectrified_pop:,}",
            f"{unelectrified_pop_pct:.1f}% of total",
            delta_color="inverse",
            help="Population in unelectrified settlements"
        )
    
    # Regional breakdown
    st.subheader("📊 Regional Electrification Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional statistics table
        regional_stats = dre_atlas_df.groupby('admin_cgaz_1').agg({
            'has_nightlight': ['sum', 'count']
        })
        regional_stats.columns = ['Electrified', 'Total']
        regional_stats['Unelectrified'] = regional_stats['Total'] - regional_stats['Electrified']
        regional_stats['Electrification Rate (%)'] = (regional_stats['Electrified'] / regional_stats['Total'] * 100).round(1)
        regional_stats = regional_stats.sort_values('Electrification Rate (%)', ascending=False)
        
        st.dataframe(regional_stats, use_container_width=True)
    
    with col2:
        # Electrification rate by region chart
        fig_region = px.bar(
            regional_stats.reset_index(),
            x='admin_cgaz_1',
            y='Electrification Rate (%)',
            title="Electrification Rate by Region",
            color='Electrification Rate (%)',
            color_continuous_scale='RdYlGn',
            labels={'admin_cgaz_1': 'Region'}
        )
        fig_region.update_layout(height=400)
        st.plotly_chart(fig_region, use_container_width=True)
    
    # Distribution analysis
    st.subheader("📈 Settlement Characteristics Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Population distribution
        fig_pop = go.Figure()
        fig_pop.add_trace(go.Box(
            y=electrified['population'],
            name='Electrified',
            marker_color='green'
        ))
        fig_pop.add_trace(go.Box(
            y=unelectrified['population'],
            name='Unelectrified',
            marker_color='red'
        ))
        fig_pop.update_layout(
            title="Population Distribution by Electrification Status",
            yaxis_title="Population",
            yaxis_type="log",
            height=400
        )
        st.plotly_chart(fig_pop, use_container_width=True)
    
    with col2:
        # Distance to grid distribution
        if 'distance_to_existing_transmission_lines' in dre_atlas_df.columns:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Box(
                y=electrified['distance_to_existing_transmission_lines'],
                name='Electrified',
                marker_color='green'
            ))
            fig_dist.add_trace(go.Box(
                y=unelectrified['distance_to_existing_transmission_lines'],
                name='Unelectrified',
                marker_color='red'
            ))
            fig_dist.update_layout(
                title="Distance to Grid by Electrification Status",
                yaxis_title="Distance (km)",
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # Infrastructure access comparison
    st.subheader("🏗️ Infrastructure Access Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Road access
        road_electrified = electrified['main_road_access'].mean() * 100
        road_unelectrified = unelectrified['main_road_access'].mean() * 100
        
        fig_road = go.Figure(data=[
            go.Bar(name='Electrified', x=['Road Access'], y=[road_electrified], marker_color='green'),
            go.Bar(name='Unelectrified', x=['Road Access'], y=[road_unelectrified], marker_color='red')
        ])
        fig_road.update_layout(
            title="Main Road Access (%)",
            yaxis_title="Percentage with Access",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_road, use_container_width=True)
    
    with col2:
        # Education facilities
        edu_electrified = electrified['has_education_facility'].mean() * 100
        edu_unelectrified = unelectrified['has_education_facility'].mean() * 100
        
        fig_edu = go.Figure(data=[
            go.Bar(name='Electrified', x=['Education'], y=[edu_electrified], marker_color='green'),
            go.Bar(name='Unelectrified', x=['Education'], y=[edu_unelectrified], marker_color='red')
        ])
        fig_edu.update_layout(
            title="Education Facilities (%)",
            yaxis_title="Percentage with Facility",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_edu, use_container_width=True)
    
    with col3:
        # Health facilities
        health_electrified = electrified['has_health_facility'].mean() * 100
        health_unelectrified = unelectrified['has_health_facility'].mean() * 100
        
        fig_health = go.Figure(data=[
            go.Bar(name='Electrified', x=['Health'], y=[health_electrified], marker_color='green'),
            go.Bar(name='Unelectrified', x=['Health'], y=[health_unelectrified], marker_color='red')
        ])
        fig_health.update_layout(
            title="Health Facilities (%)",
            yaxis_title="Percentage with Facility",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_health, use_container_width=True)
    
    # Summary insights
    st.markdown("""
    <div class="pre-step-info">
    <h4>📋 Key Insights from Current Status</h4>
    <ul>
        <li><strong>{:,} settlements</strong> remain unelectrified, representing <strong>{:.1f}%</strong> of all settlements</li>
        <li><strong>{:,} people</strong> lack electricity access (<strong>{:.1f}%</strong> of total population)</li>
        <li>Unelectrified settlements have <strong>{:.1f}x</strong> less road access compared to electrified ones</li>
        <li>Average distance to grid for unelectrified settlements: <strong>{:.1f} km</strong></li>
        <li>Regions with lowest electrification require priority attention</li>
    </ul>
    </div>
    """.format(
        unelectrified_count, 
        unelectrified_rate,
        unelectrified_pop,
        unelectrified_pop_pct,
        road_electrified / max(road_unelectrified, 1),
        unelectrified['distance_to_existing_transmission_lines'].mean() if 'distance_to_existing_transmission_lines' in unelectrified.columns else 0
    ), unsafe_allow_html=True)
    
    return unelectrified

def create_step2_interface(total_cost):
    """Create Step 2 interface for distributing total cost across years as percentages"""
    
    st.markdown("""
    <div class="step-complete">
    <h4>✅ Step 1 Complete: Total Investment Required = ${:.2f} Billion</h4>
    <p>Now distribute this investment across years 2025-2030 as percentages.</p>
    </div>
    """.format(total_cost/1e9), unsafe_allow_html=True)

    st.write("***!! Before moving to the next action, scroll down the model parameters option bar and adjust Prioritization Strategy parameters.***")
    
    st.write("**Choose how to distribute the total investment as percentages across years:**")
    
    # Distribution strategy options
    distribution_option = st.radio(
        "Distribution Strategy",
        ["Equal Distribution (16.7% per year)", "Investment Patterns", "Custom Distribution"],
        help="Choose how to spread the ${:.2f}B investment over 6 years (2025-2030)".format(total_cost/1e9)
    )
    
    years = list(range(2025, 2031))  # 6 years: 2025-2030
    annual_percentages = {}
    
    if distribution_option == "Equal Distribution (16.7% per year)":
        equal_percentage = 100.0 / len(years)  # 16.67% per year
        for year in years:
            annual_percentages[year] = equal_percentage
        st.info(f"Equal distribution: {equal_percentage:.1f}% per year")
    
    elif distribution_option == "Investment Patterns":
        pattern = st.selectbox(
            "Select Investment Pattern",
            ["Ramp-up Strategy (Gradual acceleration)", 
             "Front-loaded Strategy (Heavy initial investment)", 
             "Back-loaded Strategy (Building momentum)",
             "Focused Strategy (Peak in middle years)"],
            help="Common investment timing patterns used in infrastructure projects"
        )
        
        if pattern == "Ramp-up Strategy (Gradual acceleration)":
            percentages = [10, 12, 15, 18, 22, 23]  # Gradual increase
            st.info("Gradual acceleration: Start slow, build capacity, accelerate implementation")
        elif pattern == "Front-loaded Strategy (Heavy initial investment)":
            percentages = [30, 25, 18, 12, 10, 5]  # Heavy start
            st.info("Front-loaded: Heavy initial investment, then maintain existing infrastructure")
        elif pattern == "Back-loaded Strategy (Building momentum)":
            percentages = [8, 10, 12, 18, 25, 27]  # Heavy finish  
            st.info("Back-loaded: Build capacity and expertise, then accelerate in later years")
        else:  # Focused Strategy
            percentages = [12, 18, 25, 25, 15, 5]  # Peak in middle years
            st.info("Focused: Concentrate efforts in middle years for maximum efficiency")
        
        for i, year in enumerate(years):
            annual_percentages[year] = percentages[i]
    
    else:  # Custom Distribution
        st.write("**Set percentage of total investment for each year:**")
        
        cols = st.columns(len(years))
        for i, year in enumerate(years):
            with cols[i]:
                percentage = st.number_input(
                    f"{year}",
                    min_value=0.0, max_value=50.0, 
                    value=16.7, step=1.0,
                    key=f"pct_{year}",
                    help=f"% of ${total_cost/1e9:.2f}B to invest in {year}"
                )
                annual_percentages[year] = percentage
    
    # Validate percentages sum to 100%
    total_percentage = sum(annual_percentages.values())
    
    if abs(total_percentage - 100.0) > 0.1:
        st.error(f"⚠️ Percentages must sum to 100%. Current total: {total_percentage:.1f}%")
        if st.button("🔄 Auto-normalize to 100%", help="Automatically adjust percentages to sum to 100%"):
            # Normalize percentages to sum to 100%
            for year in years:
                annual_percentages[year] = annual_percentages[year] * (100.0 / total_percentage)
            st.rerun()
        return None
    else:
        st.success(f"✅ Distribution valid: {total_percentage:.1f}%")
    
    # Calculate annual budgets from percentages
    annual_budgets = {}
    for year in years:
        annual_budgets[year] = total_cost * (annual_percentages[year] / 100.0)
    
    # Display distribution summary
    st.subheader("📊 Investment Distribution Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution table
        distribution_data = []
        for year in years:
            distribution_data.append({
                'Year': year,
                'Percentage': f"{annual_percentages[year]:.1f}%",
                'Annual Budget': f"${annual_budgets[year]/1e9:.2f}B"
            })
        
        df_distribution = pd.DataFrame(distribution_data)
        st.dataframe(df_distribution, use_container_width=True, hide_index=True)
    
    with col2:
        # Visualization of distribution
        fig_dist = px.bar(
            x=years, 
            y=[annual_percentages[year] for year in years],
            title="Investment Distribution by Year",
            labels={'x': 'Year', 'y': 'Percentage of Total Investment (%)'},
            color=[annual_percentages[year] for year in years],
            color_continuous_scale='Blues'
        )
        fig_dist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    return annual_budgets

def create_optimized_map(settlements_df, zoom_level="medium"):
    """Create optimized map for large datasets using clustering and sampling"""
    
    # Determine sampling rate based on zoom level
    sampling_rates = {
        "low": 0.02,    # 2% of data (for overview)
        "medium": 0.1,   # 10% of data 
        "high": 0.5     # 50% of data (for detailed view)
    }
    
    sample_rate = sampling_rates.get(zoom_level, 0.1)
    
    # Sample data to improve performance
    if len(settlements_df) > 1000:
        sample_size = max(1000, int(len(settlements_df) * sample_rate))
        sample_df = settlements_df.sample(n=min(sample_size, len(settlements_df)), random_state=42)
    else:
        sample_df = settlements_df
    
    # Color mapping for technologies
    tech_colors = {
        'grid': [0, 0, 255, 160],      # Blue
        'minigrid': [0, 255, 0, 160],  # Green
        'shs': [255, 165, 0, 160],     # Orange
        '': [255, 0, 0, 160]           # Red for not electrified
    }
    
    # Prepare data for pydeck
    sample_df = sample_df.copy()
    
    # Handle color mapping properly
    def get_color(tech):
        return tech_colors.get(tech, [255, 0, 0, 160])  # Default to red for unknown
    
    sample_df['color'] = sample_df['technology_chosen'].apply(get_color)
    sample_df['size'] = sample_df['population'] / 100  # Scale point size by population
    sample_df['size'] = sample_df['size'].clip(lower=20, upper=200)  # Limit size range
    
    # Create pydeck map for better performance
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=sample_df,
        get_position='[lon, lat]',
        get_color='color',
        get_radius='size',
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=10,
    )
    
    # Set the viewport location
    view_state = pdk.ViewState(
        latitude=9.1450,
        longitude=40.4897,
        zoom=5,
        bearing=0,
        pitch=0
    )
    
    # Create the deck
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{village_name}</b><br/>"
                   "Population: {population}<br/>"
                   "Technology: {technology_chosen}<br/>"
                   "Year: {electrification_year}<br/>"
                   "Cost/Connection: ${cost_per_connection:.0f}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    )
    
    return r, len(sample_df), len(settlements_df)

def create_budget_dashboard(yearly_results, annual_budgets):
    """Create budget tracking dashboard with flexible budgets"""
    if not yearly_results:
        return None
        
    df = pd.DataFrame(yearly_results)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Annual Spending vs Budget', 'Cumulative Carbon Revenue', 
                       'Cost Effectiveness', 'Budget Utilization'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Annual spending vs budget
    fig.add_trace(
        go.Bar(x=df['year'], y=df['total_spent']/1e6, name='Spent', marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df['year'], y=df['budget']/1e6, name='Budget', marker_color='lightcoral', opacity=0.7),
        row=1, col=1
    )
    
    # Cumulative carbon revenue
    df['cumulative_carbon_revenue'] = df['carbon_revenue'].cumsum()
    fig.add_trace(
        go.Scatter(x=df['year'], y=df['cumulative_carbon_revenue']/1e6, 
                  mode='lines+markers', name='Carbon Revenue', line=dict(color='green')),
        row=1, col=2
    )
    
    # Cost effectiveness ($/connection)
    df['cost_per_connection'] = df['total_spent'] / df['settlements_electrified'].replace(0, 1)
    fig.add_trace(
        go.Bar(x=df['year'], y=df['cost_per_connection'], name='Cost/Connection', marker_color='orange'),
        row=2, col=1
    )
    
    # Budget utilization
    df['budget_utilization'] = (df['total_spent'] + df['carbon_revenue']) / df['budget'] * 100
    fig.add_trace(
        go.Bar(x=df['year'], y=df['budget_utilization'], name='Utilization %', marker_color='purple'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Budget Dashboard")
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="USD Million", row=1, col=1)
    fig.update_yaxes(title_text="USD Million", row=1, col=2)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_yaxes(title_text="Percent", row=2, col=2)
    
    return fig

def create_electrification_progress_chart(yearly_results):
    """Create electrification progress visualization"""
    if not yearly_results:
        return None
        
    df = pd.DataFrame(yearly_results)
    
    fig = go.Figure()
    
    # Electrification rate over time
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['electrification_rate'],
        mode='lines+markers',
        name='Electrification Rate',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add target line at 100%
    fig.add_hline(y=100, line_dash="dash", line_color="red", 
                  annotation_text="100% Target")
    
    fig.update_layout(
        title="Ethiopia Electrification Progress (2025-2030)",
        xaxis_title="Year",
        yaxis_title="Electrification Rate (%)",
        height=400,
        showlegend=True
    )
    
    return fig

def create_full_cost_analysis_dashboard(full_cost_results):
    """Create dashboard showing full electrification cost analysis"""
    if not full_cost_results:
        return None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Investment Required",
            f"${full_cost_results['total_cost']/1e9:.2f}B",
            help="Total cost to electrify all unelectrified settlements"
        )
    
    with col2:
        st.metric(
            "Total Connections",
            f"{full_cost_results['total_connections']:,}",
            help="Number of connections to be made"
        )
    
    with col3:
        st.metric(
            "Average Cost per Connection",
            f"${full_cost_results['average_cost_per_connection']:.0f}",
            help="Average investment per connection"
        )
    
    with col4:
        # Timeline: 2025-2030 (6 years)
        st.metric(
            "Implementation Timeline",
            "6 years",
            help="Timeline: 2025-2030 (6 years)"
        )
    
    # Technology breakdown charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Technology breakdown by connections
        tech_breakdown = full_cost_results['technology_breakdown']
        fig_tech = px.pie(
            values=list(tech_breakdown.values()),
            names=list(tech_breakdown.keys()),
            title="Technology Mix by Connections",
            color_discrete_map={
                'grid': '#1f77b4',      # Blue
                'minigrid': '#2ca02c',  # Green  
                'shs': '#ff7f0e'        # Orange
            }
        )
        st.plotly_chart(fig_tech, use_container_width=True)
    
    with col2:
        # Cost breakdown by technology
        cost_breakdown = full_cost_results['cost_breakdown']
        fig_cost = px.pie(
            values=list(cost_breakdown.values()),
            names=list(cost_breakdown.keys()),
            title="Investment by Technology",
            color_discrete_map={
                'grid': '#1f77b4',      # Blue
                'minigrid': '#2ca02c',  # Green
                'shs': '#ff7f0e'        # Orange
            }
        )
        st.plotly_chart(fig_cost, use_container_width=True)

def main():
    # Header
    st.title("⚡ Ethiopia Electrification Decision Support Tool")
    with st.expander("ℹ️ About this Tool – Click to Expand"):
        st.markdown(
            """
            ### Complete Electrification Analysis
            This tool is an **energy modeling and decision-support platform** built to explore Ethiopia’s 
            pathways to achieving **100% electrification by 2030**. It combines the World Bank's DRE-ATLAS data, user defined technology 
            cost models, and financing scenarios into an interactive Streamlit application that allows 
            policymakers, planners, and researchers to test different electrification strategies.

            ---
            #### ⚙️ How it Works

            **Pre-Step – Electrification Status Check**  
            - Uses satellite nightlight data from the DRE atlas to identify which settlements are already electrified.  
            - Separates settlements into *electrified* and *unelectrified*.  

            **Step 1 – Full Electrification Cost Calculation**  
            - Calculates the **total cost** of providing electricity to all unelectrified settlements without budget constraints. The cost of the technology is defined by the user.
            - The technology for a settlement is selected based on settlement characteristics(road access, distance from existing grid, population size)
            **Step 2 – Budget-Constrained Optimization with**  
            - Uses the Step 1 total cost as the baseline.  
            - Lets the user distribute the total cost as **percentages across 2025–2030** to simulate phased electrification.  
            -The user gets to adjust weights to balance competing objectives for prioritizing a settlement (
            - Generates maps and tables showing rollout progress, budget use, and technology mix.  
            """
        )
    
    # Load data
    with st.spinner("Loading settlement data..."):
        dre_atlas_df, unelectrified_df = load_and_prepare_data()
    
    # Sidebar parameters
    params = create_parameter_sidebar()
    
    # Main tabs including new Pre-Step
    tab_prestep, tab_step1, tab_step2, tab_strategy, tab_dashboard, tab_investment, tab_map, tab_progress, tab_carbon, tab_details = st.tabs([
        "🔌 Pre-Step: Current Status", "🎯 Step 1: Calculate Investment", "📊 Step 2: Distribute & Simulate",
        "⚖️ Strategy Analysis", "📊 Dashboard", "💰 Investment Analysis", "🗺️ Settlement Map", 
        "📈 Progress", "🌱 Carbon Impact", "📋 Details"
    ])
    
    # Pre-Step Tab
    with tab_prestep:
        unelectrified_from_analysis = create_prestep_analysis(dre_atlas_df)
        
        # Show data summary
        st.markdown("---")
      #  st.subheader("📁 Data Files Status")
        #col1, col2 = st.columns(2)
       # with col1:
       #     st.info(f"✅ **dre_atlas.csv** loaded: {len(dre_atlas_df):,} total settlements")
       # with col2:
        #    st.info(f"✅ **unelectrified.csv** ready: {len(unelectrified_df):,} unelectrified settlements")
    
    # Step 1 Tab - ENHANCED WITH TECHNOLOGY DISTRIBUTION
    with tab_step1:
        st.header("🎯 Step 1: Calculate Total Investment Required")
        st.write("**Calculate the total cost to electrify ALL unelectrified settlements (without budget constraints)**")
        st.write("***- Adjust model parameters :(Technology costs, carbon finance, Grid mix)***")

        st.info(f"📊 Working with {len(unelectrified_df):,} unelectrified settlements ")
        
        # Step 1 button and results
        step1_complete = 'full_cost_results' in st.session_state
        
        if not step1_complete:
            if st.button("🔍 Run Step 1: Calculate Total Investment Required", type="primary", use_container_width=True):
                with st.spinner(f"Calculating total electrification cost for {len(unelectrified_df):,} unelectrified settlements..."):
                    model = ElectrificationModel(unelectrified_df)
                    full_cost_results = model.calculate_full_electrification_cost(params)
                    st.session_state['full_cost_results'] = full_cost_results
                    st.session_state['params'] = params
                    st.rerun()
        else:
            # Show Step 1 results
            full_cost_results = st.session_state['full_cost_results']
            
            st.markdown("""
            <div class="step-complete">
            <h4>✅ Step 1 Complete: Full Electrification Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Investment Required", f"${full_cost_results['total_cost']/1e9:.2f}B")
            with col2:
                st.metric("Unelectrified Settlements", f"{len(unelectrified_df):,}")
            with col3:
                st.metric("Total Connections", f"{full_cost_results['total_connections']:,}")
            with col4:
                st.metric("Average Cost/Connection", f"${full_cost_results['average_cost_per_connection']:.0f}")
            
            # Technology Distribution Analysis
            st.markdown("---")
            st.subheader("🔧 Technology Distribution Analysis")
            
            # Calculate settlement percentages by technology
            settlement_details_df = pd.DataFrame(full_cost_results['settlement_details'])
            tech_settlement_counts = settlement_details_df['technology'].value_counts()
            total_settlements = len(settlement_details_df)
            
            # Create percentage breakdown
            tech_percentages = (tech_settlement_counts / total_settlements * 100).round(1)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Settlement distribution metrics
                st.markdown("**📍 Settlements by Technology**")
                for tech in ['grid', 'minigrid', 'shs']:
                    if tech in tech_percentages.index:
                        count = tech_settlement_counts[tech]
                        pct = tech_percentages[tech]
                        st.metric(
                            f"{tech.upper()} Settlements",
                            f"{count:,}",
                            f"{pct}% of total"
                        )
            
            with col2:
                # Pie chart of settlement distribution
                fig_settlements = px.pie(
                    values=tech_settlement_counts.values,
                    names=tech_settlement_counts.index,
                    title="Settlement Distribution by Technology",
                    color_discrete_map={
                        'grid': '#1f77b4',
                        'minigrid': '#2ca02c',
                        'shs': '#ff7f0e'
                    }
                )
                fig_settlements.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>' +
                                  'Settlements: %{value:,}<br>' +
                                  'Percentage: %{percent}<br>' +
                                  '<extra></extra>'
                )
                fig_settlements.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig_settlements, use_container_width=True)
            
            with col3:
                # Connection distribution
                tech_connections = full_cost_results['technology_breakdown']
                total_connections_by_tech = sum(tech_connections.values())
                
                st.markdown("**🔌 Connections by Technology**")
                for tech in ['grid', 'minigrid', 'shs']:
                    if tech in tech_connections:
                        connections = tech_connections[tech]
                        pct = (connections / total_connections_by_tech * 100)
                        st.metric(
                            f"{tech.upper()} Connections",
                            f"{connections:,}",
                            f"{pct:.1f}% of total"
                        )
            
            # Additional analysis
            st.markdown("---")
            st.subheader("📊 Detailed Technology Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cost breakdown by technology
                cost_by_tech = full_cost_results['cost_breakdown']
                cost_df = pd.DataFrame([
                    {'Technology': tech.upper(), 'Investment (USD B)': cost/1e9, 
                     'Percentage': (cost/full_cost_results['total_cost']*100)}
                    for tech, cost in cost_by_tech.items()
                ])
                
                fig_cost = px.bar(
                    cost_df,
                    x='Technology',
                    y='Investment (USD B)',
                    title="Investment Requirements by Technology",
                    color='Investment (USD B)',
                    color_continuous_scale='Blues',
                    text='Percentage'
                )
                fig_cost.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_cost.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_cost, use_container_width=True)
            
            with col2:
                # Average cost per connection by technology
                avg_cost_by_tech = {}
                for tech in ['grid', 'minigrid', 'shs']:
                    if tech in cost_by_tech and tech in tech_connections and tech_connections[tech] > 0:
                        avg_cost_by_tech[tech] = cost_by_tech[tech] / tech_connections[tech]
                
                if avg_cost_by_tech:
                    avg_cost_df = pd.DataFrame([
                        {'Technology': tech.upper(), 'Avg Cost per Connection (USD)': cost}
                        for tech, cost in avg_cost_by_tech.items()
                    ])
                    
                    fig_avg_cost = px.bar(
                        avg_cost_df,
                        x='Technology',
                        y='Avg Cost per Connection (USD)',
                        title="Average Cost per Connection by Technology",
                        color='Avg Cost per Connection (USD)',
                        color_continuous_scale='Reds',
                        text='Avg Cost per Connection (USD)'
                    )
                    fig_avg_cost.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
                    fig_avg_cost.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_avg_cost, use_container_width=True)
            
            # Summary statistics table
            st.markdown("---")
            st.subheader("📈 Technology Selection Summary")
            
            summary_data = []
            for tech in ['grid', 'minigrid', 'shs']:
                if tech in tech_settlement_counts.index:
                    summary_data.append({
                        'Technology': tech.upper(),
                        'Settlements': f"{tech_settlement_counts[tech]:,}",
                        'Settlement %': f"{tech_percentages[tech]}%",
                        'Connections': f"{tech_connections.get(tech, 0):,}",
                        'Connection %': f"{(tech_connections.get(tech, 0)/total_connections_by_tech*100):.1f}%",
                        'Total Investment': f"${cost_by_tech.get(tech, 0)/1e9:.2f}B",
                        'Investment %': f"{(cost_by_tech.get(tech, 0)/full_cost_results['total_cost']*100):.1f}%",
                        'Avg Cost/Connection': f"${avg_cost_by_tech.get(tech, 0):.0f}" if tech in avg_cost_by_tech else "N/A"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Key insights
            st.info(f"""
            **🔍 Key Insights from Technology Distribution:**
            - **{tech_percentages.idxmax().upper()}** is selected for the most settlements ({tech_percentages.max()}%)
            - **{max(tech_connections, key=tech_connections.get).upper()}** serves the most connections ({max(tech_connections.values()):,})
            - **{max(cost_by_tech, key=cost_by_tech.get).upper()}** requires the highest investment (${max(cost_by_tech.values())/1e9:.2f}B)
            - Technology selection is based on: cost efficiency ({params.get('cost_efficiency_weight', 0.6)*100:.0f}% weight) and reliability ({params.get('reliability_weight', 0.3)*100:.0f}% weight)
            """)
            
            # Technology selection logic explanation
            with st.expander("📖 How Technology Selection Works"):
                st.markdown("""
                **The tool selects technologies based on settlement characteristics:**
                
                **🔌 Grid Extension** is preferred for:
                - Settlements within **15 km** of existing grid
                - Large settlements (>2000 population)
                - Settlements with >300 connections
                - Areas with good road access
                
                **⚡ Mini-grids** are ideal for:
                - Medium distance from grid (10-50 km)
                - Medium-sized settlements (200-2000 population)
                - Higher energy demand per connection (productive use)
                - Clustered settlement patterns
                
                **☀️ Solar Home Systems (SHS)** work best for:
                - Remote settlements (>30 km from grid)
                - Small settlements (<500 population)
                - Scattered settlements with <50 connections
                - Areas without road access
                
                **Special Cases:**
                - Very close + large settlements (≤5km, >3000 pop) → Always Grid
                - Very remote + tiny settlements (>75km, <20 connections) → Always SHS
                - Ideal mini-grid conditions (15-40km, 500-1500 pop, high demand) → Always Mini-grid
                
               
                """)
            
            if st.button("🔄 Recalculate Step 1", help="Recalculate with new parameters"):
                del st.session_state['full_cost_results']
                if 'yearly_results' in st.session_state:
                    del st.session_state['yearly_results']
                st.rerun()
    
    # Step 2 Tab
    with tab_step2:
        st.header("📊 Step 2: Distribute Investment Across Years (2025-2030)")
        
        step1_complete = 'full_cost_results' in st.session_state
        
        if not step1_complete:
            st.markdown("""
            <div class="step-pending">
            <h4>⚠️ Step 2 requires Step 1 to be completed first</h4>
            <p>Please run Step 1 to calculate the total investment required.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Step 2 interface
            total_cost = st.session_state['full_cost_results']['total_cost']
            annual_budgets = create_step2_interface(total_cost)
            
            if annual_budgets is not None:
                if st.button("🚀 Run Step 2: Phased Implementation Simulation", type="primary", use_container_width=True):
                    with st.spinner(f"Running phased implementation simulation for {len(unelectrified_df):,} settlements..."):
                        model = ElectrificationModel(unelectrified_df)
                        yearly_results = model.simulate_electrification(params, annual_budgets, start_year=2025)
                        settlements_df = model.settlements
                        
                        # Store results in session state
                        st.session_state['yearly_results'] = yearly_results
                        st.session_state['settlements_df'] = settlements_df
                        st.session_state['annual_budgets'] = annual_budgets
                        
                        final_rate = yearly_results[-1]['electrification_rate'] if yearly_results else 0
                        total_achieved = settlements_df['electrified'].sum()
                        
                        # Show immediate results
                        st.markdown("""
                        <div class="step-complete">
                        <h4>✅ Step 2 Complete: Phased Implementation Results</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Final Electrification Rate", f"{final_rate:.1f}%")
                        with col2:
                            st.metric("Settlements Electrified", f"{total_achieved:,}")
                        with col3:
                            investment_used = sum(r['total_spent'] for r in yearly_results) / 1e9
                            st.metric("Investment Used", f"${investment_used:.2f}B")
    
    # Strategy Analysis Tab
    with tab_strategy:
        st.header("⚖️ Prioritization Strategy Analysis")
        
        if 'settlements_df' in st.session_state and 'yearly_results' in st.session_state:
            settlements_df = st.session_state['settlements_df']
            
            # Display current strategy weights
            st.subheader("🎯 Current Prioritization Weights")
            
            weights_data = [
                {"Factor": "Population Scale", "Weight": params['population_weight'], 
                 "Description": "Prioritize larger settlements to serve more people"},
                {"Factor": "Poverty Focus", "Weight": params['poverty_weight'], 
                 "Description": "Prioritize poor communities with low wealth index"},
                {"Factor": "Cost Efficiency", "Weight": params['cost_efficiency_weight'], 
                 "Description": "Prioritize settlements with lower cost per connection"},
                {"Factor": "Implementation Speed", "Weight": params['implementation_ease_weight'], 
                 "Description": "Prioritize settlements with road access (easier to build)"},
                {"Factor": "Social Infrastructure", "Weight": params['social_infrastructure_weight'], 
                 "Description": "Prioritize settlements with schools/health facilities"},
                {"Factor": "Technology Reliability", "Weight": params['reliability_weight'], 
                 "Description": "Prefer more reliable technologies (grid/minigrid over SHS)"},
                {"Factor": "Productive Use", "Weight": params['productive_use_weight'], 
                 "Description": "Prioritize settlements with commercial/agricultural potential"},
                {"Factor": "Security", "Weight": params['security_weight'], 
                 "Description": "Prioritize secure areas, avoid high-risk zones"}
            ]
            
            # Create weight visualization
            weights_df = pd.DataFrame(weights_data)
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Table view
                st.dataframe(weights_df[["Factor", "Weight"]], use_container_width=True)
                
                # Strategy summary
                max_weight_row = weights_df.loc[weights_df['Weight'].idxmax()]
                st.info(f"**Primary Focus:** {max_weight_row['Factor']} ({max_weight_row['Weight']:.0%})")
            
            with col2:
                # Radar chart of weights
                fig_weights = px.bar(
                    weights_df,
                    x='Weight',
                    y='Factor',
                    orientation='h',
                    title="Prioritization Weight Distribution",
                    color='Weight',
                    color_continuous_scale='Blues',
                    hover_data=['Description']
                )
                fig_weights.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_weights, use_container_width=True)
            
            # Impact Analysis
            st.subheader("📊 Strategy Impact Analysis")
            
            # Get electrified settlements
            electrified = settlements_df[settlements_df['electrified'] == True].copy()
            unelectrified = settlements_df[settlements_df['electrified'] == False].copy()
            
            if len(electrified) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Population impact
                    avg_pop_electrified = electrified['population'].mean()
                    avg_pop_unelectrified = unelectrified['population'].mean() if len(unelectrified) > 0 else 0
                    
                    st.metric(
                        "Avg Population (Electrified)",
                        f"{avg_pop_electrified:.0f}",
                        f"{((avg_pop_electrified/avg_pop_unelectrified - 1) * 100):.1f}% vs unelectrified" if avg_pop_unelectrified > 0 else "N/A"
                    )
                
                with col2:
                    # Poverty impact
                    avg_rwi_electrified = electrified['mean_rwi'].mean()
                    avg_rwi_unelectrified = unelectrified['mean_rwi'].mean() if len(unelectrified) > 0 else 0
                    
                    st.metric(
                        "Avg Wealth Index (Electrified)",
                        f"{avg_rwi_electrified:.2f}",
                        f"{'Poorer' if avg_rwi_electrified < avg_rwi_unelectrified else 'Wealthier'} than unelectrified",
                        delta_color="inverse" if avg_rwi_electrified < avg_rwi_unelectrified else "normal"
                    )
                
                with col3:
                    # Cost efficiency
                    avg_cost = electrified['cost_per_connection'].mean()
                    st.metric(
                        "Avg Cost per Connection",
                        f"${avg_cost:.0f}",
                        help="Average cost achieved through prioritization"
                    )
                
                # Technology distribution based on strategy
                st.subheader("🔧 Technology Mix Impact")
                
                tech_by_priority = electrified.groupby('technology_chosen').agg({
                    'priority_score': 'mean',
                    'geohash': 'count',
                    'population': 'sum'
                }).rename(columns={'geohash': 'count', 'priority_score': 'avg_priority'})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_tech_priority = px.bar(
                        tech_by_priority.reset_index(),
                        x='technology_chosen',
                        y='avg_priority',
                        title="Average Priority Score by Technology",
                        color='avg_priority',
                        color_continuous_scale='Viridis'
                    )
                    fig_tech_priority.update_layout(height=300)
                    st.plotly_chart(fig_tech_priority, use_container_width=True)
                
                with col2:
                    fig_tech_count = px.pie(
                        values=tech_by_priority['count'],
                        names=tech_by_priority.index,
                        title="Technology Distribution (Count)"
                    )
                    fig_tech_count.update_layout(height=300)
                    st.plotly_chart(fig_tech_count, use_container_width=True)
                
                # Top prioritized settlements
                st.subheader("🏆 Top 10 Prioritized Settlements")
                
                if 'priority_score' in electrified.columns:
                    top_settlements = electrified.nlargest(10, 'priority_score')[
                        ['village_name', 'admin_cgaz_1', 'population', 'technology_chosen', 
                         'cost_per_connection', 'priority_score', 'electrification_year']
                    ]
                    
                    st.dataframe(
                        top_settlements.style.format({
                            'population': '{:,.0f}',
                            'cost_per_connection': '${:,.0f}',
                            'priority_score': '{:.3f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Strategy effectiveness metrics
                    st.subheader("📈 Strategy Effectiveness")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Social impact
                        edu_coverage = electrified['has_education_facility'].mean() * 100
                        health_coverage = electrified['has_health_facility'].mean() * 100
                        
                        fig_social = go.Figure(data=[
                            go.Bar(name='Education', x=['Coverage'], y=[edu_coverage]),
                            go.Bar(name='Health', x=['Coverage'], y=[health_coverage])
                        ])
                        fig_social.update_layout(
                            title="Social Infrastructure Coverage (%)",
                            yaxis_title="Percentage",
                            height=300
                        )
                        st.plotly_chart(fig_social, use_container_width=True)
                    
                    with col2:
                        # Security distribution
                        security_dist = electrified['security_risk'].value_counts()
                        fig_security = px.pie(
                            values=security_dist.values,
                            names=security_dist.index,
                            title="Security Risk Distribution",
                            color_discrete_map={'low': 'green', 'medium': 'yellow', 'high': 'red'}
                        )
                        fig_security.update_layout(height=300)
                        st.plotly_chart(fig_security, use_container_width=True)
            else:
                st.info("Run Step 2 simulation to see strategy impact analysis")
        else:
            st.info("⚖️ Complete Step 2 to see strategy analysis!")
    
    # Dashboard Tab
    with tab_dashboard:
        st.header("📊 Electrification Dashboard")
        
        if 'yearly_results' in st.session_state:
            yearly_results = st.session_state['yearly_results']
            settlements_df = st.session_state['settlements_df']
            full_cost_results = st.session_state['full_cost_results']
            
            # Main progress chart
            st.subheader("📈 Electrification Progress (2025-2030)")
            progress_fig = create_electrification_progress_chart(yearly_results)
            if progress_fig:
                st.plotly_chart(progress_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Investment comparison
                st.subheader("💰 Investment Analysis")
                total_invested = sum(r['total_spent'] for r in yearly_results) / 1e9
                total_required = full_cost_results['total_cost'] / 1e9
                
                fig_investment = go.Figure()
                fig_investment.add_trace(go.Bar(
                    x=['Used', 'Remaining'],
                    y=[total_invested, max(0, total_required - total_invested)],
                    marker_color=['lightblue', 'lightgray'],
                    text=[f"${total_invested:.2f}B", f"${max(0, total_required - total_invested):.2f}B"],
                    textposition='auto'
                ))
                fig_investment.update_layout(
                    title="Investment Utilization",
                    yaxis_title="USD Billions",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig_investment, use_container_width=True)
            
            with col2:
                # Technology breakdown
                st.subheader("🔧 Technology Mix")
                tech_counts = settlements_df[settlements_df['electrified']]['technology_chosen'].value_counts()
                
                if not tech_counts.empty:
                    fig_pie = px.pie(
                        values=tech_counts.values, 
                        names=tech_counts.index, 
                        title="Technology Distribution"
                    )
                    fig_pie.update_layout(height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No electrification data available yet.")
        else:
            st.info("📊 Complete Steps 1 and 2 to see dashboard results!")
    
    # Investment Analysis Tab
    with tab_investment:
        st.header("💰 Investment Analysis")
        
        if 'full_cost_results' in st.session_state:
            full_cost_results = st.session_state['full_cost_results']
            create_full_cost_analysis_dashboard(full_cost_results)
            
            # If Step 2 is also complete, show comparison
            if 'yearly_results' in st.session_state:
                st.subheader("🎯 Implementation vs. Full Requirement")
                
                yearly_results = st.session_state['yearly_results']
                achieved_investment = sum([r['total_spent'] for r in yearly_results])
                achieved_rate = yearly_results[-1]['electrification_rate'] if yearly_results else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    coverage = (achieved_investment / full_cost_results['total_cost']) * 100
                    st.metric("Budget Coverage", f"{coverage:.1f}%")
                
                with col2:
                    efficiency = achieved_rate / coverage if coverage > 0 else 0
                    st.metric("Electrification Efficiency", f"{efficiency:.2f}",
                             help="Electrification rate achieved per % of budget used")
                
                with col3:
                    remaining_cost = (full_cost_results['total_cost'] - achieved_investment) / 1e9
                    st.metric("Remaining Investment", f"${remaining_cost:.2f}B")
        else:
            st.info("💰 Complete Step 1 to see investment analysis!")
    
    # Map Tab
    with tab_map:
        st.header("🗺️ Settlement Map")
        
        if 'settlements_df' in st.session_state:
            settlements_df = st.session_state['settlements_df']
            
            # Map controls
            col1, col2, col3 = st.columns(3)
            with col1:
                zoom_level = st.selectbox("Map Detail Level", 
                                        options=['low', 'medium', 'high'],
                                        index=1,
                                        help="Higher detail shows more points but may be slower")
            with col2:
                show_tech = st.selectbox("Filter Technology", 
                                       options=['All', 'grid', 'minigrid', 'shs', 'Not Electrified'])
            with col3:
                show_year = st.selectbox("Show Year", 
                                       options=['All'] + list(range(2025, 2031)),
                                       index=0)
            
            # Filter data
            map_data = settlements_df.copy()
            
            if show_tech != 'All':
                if show_tech == 'Not Electrified':
                    map_data = map_data[~map_data['electrified']]
                else:
                    map_data = map_data[map_data['technology_chosen'] == show_tech]
            
            if show_year != 'All':
                map_data = map_data[(map_data['electrification_year'] == show_year) | 
                                   (map_data['electrification_year'] == 0)]
            
            # Create optimized map
            deck_map, shown_points, total_points = create_optimized_map(map_data, zoom_level)
            st.pydeck_chart(deck_map)
            
            # Map info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points Shown", f"{shown_points:,}")
            with col2:
                st.metric("Total Points", f"{total_points:,}")
            with col3:
                st.metric("Sampling Rate", f"{(shown_points/max(total_points,1)*100):.1f}%")
            with col4:
                electrified_shown = map_data['electrified'].sum() if len(map_data) > 0 else 0
                st.metric("Electrified Shown", f"{electrified_shown:,}")
        else:
            st.info("🗺️ Complete Step 2 to see settlement map!")
    
    # Progress Tab
    with tab_progress:
        st.header("📈 Implementation Progress Analysis")
        
        if 'yearly_results' in st.session_state:
            yearly_results = st.session_state['yearly_results']
            annual_budgets = st.session_state['annual_budgets']
            
            budget_fig = create_budget_dashboard(yearly_results, annual_budgets)
            if budget_fig:
                st.plotly_chart(budget_fig, use_container_width=True)
            
            # Implementation execution table
            st.subheader("📅 Annual Implementation Summary")
            budget_df = pd.DataFrame(yearly_results)
            budget_df['Planned (USD B)'] = budget_df['budget'] / 1e9
            budget_df['Executed (USD B)'] = budget_df['total_spent'] / 1e9
            budget_df['Carbon Revenue (USD M)'] = budget_df['carbon_revenue'] / 1e6
            budget_df['Execution Rate (%)'] = (budget_df['total_spent'] / budget_df['budget'] * 100).round(1)
            budget_df['Cumulative Electrification (%)'] = budget_df['electrification_rate'].round(1)
            
            display_cols = ['year', 'Planned (USD B)', 'Executed (USD B)', 'Carbon Revenue (USD M)', 
                          'settlements_electrified', 'Execution Rate (%)', 'Cumulative Electrification (%)']
            st.dataframe(budget_df[display_cols], use_container_width=True)
        else:
            st.info("📈 Complete Step 2 to see progress analysis!")
    
    # Carbon Impact Tab
    with tab_carbon:
        st.header("🌱 Carbon Impact Analysis")
        
        if 'yearly_results' in st.session_state:
            yearly_results = st.session_state['yearly_results']
            
            # Carbon metrics
            col1, col2, col3 = st.columns(3)
            total_carbon_revenue = sum([r['carbon_revenue'] for r in yearly_results]) / 1e6
            annual_co2_avoided = sum([r['co2_avoided'] for r in yearly_results])
            total_investment = sum([r['total_spent'] for r in yearly_results]) / 1e6
            
            with col1:
                st.metric("Total Carbon Revenue", f"${total_carbon_revenue:.1f}M")
            with col2:
                st.metric("Annual CO₂ Avoided", f"{annual_co2_avoided/1000:.1f}K tons")
            with col3:
                carbon_roi = (total_carbon_revenue / max(total_investment, 1)) * 100
                st.metric("Carbon ROI", f"{carbon_roi:.1f}%")
            
            # Carbon impact over time
            df = pd.DataFrame(yearly_results)
            
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Annual CO₂ Avoided', 'Cumulative Carbon Revenue'))
            
            fig.add_trace(
                go.Bar(x=df['year'], y=df['co2_avoided']/1000, name='CO₂ Avoided (K tons)', 
                      marker_color='green'),
                row=1, col=1
            )
            
            df['cumulative_carbon_revenue'] = df['carbon_revenue'].cumsum()
            fig.add_trace(
                go.Scatter(x=df['year'], y=df['cumulative_carbon_revenue']/1e6, 
                          mode='lines+markers', name='Carbon Revenue (USD M)',
                          line=dict(color='blue')),
                row=1, col=2
            )
            
            fig.update_layout(height=400, title="Carbon Impact Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🌱 Complete Step 2 to see carbon impact analysis!")
    
    # Settlement Details Tab
    with tab_details:
        st.header("📋 Settlement Details")
        
        # Choose which dataset to display
        data_source = st.radio(
            "Select Data Source",
            ["All Settlements (dre_atlas.csv)", "Simulation Results (unelectrified.csv)"],
            help="View all settlements or only simulation results"
        )
        
        if data_source == "All Settlements (dre_atlas.csv)":
            display_df = dre_atlas_df
            st.info(f"📊 Showing all {len(display_df):,} settlements from dre_atlas.csv")
            
            # Filters for all settlements
            col1, col2, col3 = st.columns(3)
            with col1:
                electrification_filter = st.selectbox(
                    "Electrification Status",
                    ['All', 'Electrified (has nightlight)', 'Unelectrified (no nightlight)']
                )
            with col2:
                region_filter = st.selectbox("Region", ['All'] + list(display_df['admin_cgaz_1'].unique()))
            with col3:
                min_pop = st.number_input("Min Population", value=0, step=100)
            
            # Apply filters
            filtered_df = display_df.copy()
            
            if electrification_filter == 'Electrified (has nightlight)':
                filtered_df = filtered_df[filtered_df['has_nightlight'] == True]
            elif electrification_filter == 'Unelectrified (no nightlight)':
                filtered_df = filtered_df[filtered_df['has_nightlight'] == False]
            
            if region_filter != 'All':
                filtered_df = filtered_df[filtered_df['admin_cgaz_1'] == region_filter]
            
            filtered_df = filtered_df[filtered_df['population'] >= min_pop]
            
            # Display columns for all settlements
            display_columns = [
                'village_name', 'admin_cgaz_1', 'admin_cgaz_2', 'population', 
                'num_connections', 'has_nightlight', 'distance_to_existing_transmission_lines',
                'main_road_access', 'has_education_facility', 'has_health_facility',
                'mean_rwi', 'security_risk'
            ]
            
            # Filter to existing columns
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            
        else:  # Simulation Results
            if 'settlements_df' in st.session_state:
                display_df = st.session_state['settlements_df']
                st.info(f"📊 Showing {len(display_df):,} settlements from simulation results")
                
                # Filters for simulation results
                col1, col2, col3 = st.columns(3)
                with col1:
                    status_filter = st.selectbox("Status", ['All', 'Electrified', 'Not Electrified'])
                with col2:
                    tech_filter = st.selectbox("Technology", ['All'] + list(display_df['technology_chosen'].unique()))
                with col3:
                    region_filter = st.selectbox("Region", ['All'] + list(display_df['admin_cgaz_1'].unique()))
                
                # Apply filters
                filtered_df = display_df.copy()
                
                if status_filter == 'Electrified':
                    filtered_df = filtered_df[filtered_df['electrified']]
                elif status_filter == 'Not Electrified':
                    filtered_df = filtered_df[~filtered_df['electrified']]
                    
                if tech_filter != 'All':
                    filtered_df = filtered_df[filtered_df['technology_chosen'] == tech_filter]
                    
                if region_filter != 'All':
                    filtered_df = filtered_df[filtered_df['admin_cgaz_1'] == region_filter]
                
                # Display columns for simulation results
                display_columns = [
                    'village_name', 'admin_cgaz_1', 'population', 'num_connections',
                    'technology_chosen', 'electrification_year', 'cost_per_connection',
                    'carbon_avoided_annual', 'security_risk', 'main_road_access'
                ]
                
                # Filter to existing columns
                display_columns = [col for col in display_columns if col in filtered_df.columns]
            else:
                st.info("📋 Complete Step 2 to see simulation results!")
                return
        
        # Display the dataframe
        st.dataframe(
            filtered_df[display_columns].sort_values('population', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Settlements", len(filtered_df))
        with col2:
            st.metric("Total Population", f"{filtered_df['population'].sum():,}")
        with col3:
            if 'cost_per_connection' in filtered_df.columns:
                avg_cost = filtered_df[filtered_df['cost_per_connection'] > 0]['cost_per_connection'].mean()
                st.metric("Avg Cost/Connection", f"${avg_cost:.0f}" if not pd.isna(avg_cost) else "N/A")
            else:
                st.metric("Connections", f"{filtered_df['num_connections'].sum():,}")
        with col4:
            if 'has_nightlight' in filtered_df.columns:
                electrified_pct = filtered_df['has_nightlight'].mean() * 100
                st.metric("Electrified %", f"{electrified_pct:.1f}%")
            else:
                st.metric("Total Connections", f"{filtered_df['num_connections'].sum():,}")

if __name__ == "__main__":

    main()












