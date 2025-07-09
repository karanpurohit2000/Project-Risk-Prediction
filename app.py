import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import cohere
import joblib

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="AI Project Risk Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ML Model Loading ---
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('models/random_forest_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        return model, scaler, feature_names
    except Exception as e:
        return None, None, None

# Load ML model once at startup
ml_model, ml_scaler, feature_names = load_ml_model()

@st.cache_resource
def get_cohere_client():
    """Try to get from secrets first, fallback to env var"""
    try:
        api_key = st.secrets.get("COHERE_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv('COHERE_API_KEY')

    if api_key:
        try:
            return cohere.Client(api_key)
        except Exception as e:
            st.warning(f"Failed to initialize Cohere: {e}")
            return None
    else:
        st.warning("⚠️ COHERE_API_KEY not found in secrets or env vars")
        return None


# CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .risk-card {
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .low-risk {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.3);
    }
    
    .medium-risk {
        background: linear-gradient(135deg, #FF9800 0%, #f57c00 100%);
        color: white;
        box-shadow: 0 15px 35px rgba(255, 152, 0, 0.3);
    }
    
    .high-risk {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        box-shadow: 0 15px 35px rgba(244, 67, 54, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .improvement-suggestion {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #2196f3;
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .improvement-suggestion:hover {
        transform: translateX(10px);
    }
    
    .plotly-graph-div {
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- ML Prediction System ---
def predict_project_risk(data):
    """Pure ML-based prediction"""
    if ml_model and ml_scaler and feature_names:
        try:
            # Prepare input in correct order (using feature names from training)
            features = [data[feature] for feature in feature_names]
            
            # Scale features
            scaled_features = ml_scaler.transform([features])
            
            # Predict probabilities
            probabilities = ml_model.predict_proba(scaled_features)[0]
            
            # Convert to risk score (0-100 scale)
            risk_score = probabilities[1] * 100
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "Low Risk"
                risk_category = "low"
            elif risk_score < 60:
                risk_level = "Medium Risk"
                risk_category = "medium"
            else:
                risk_level = "High Risk"
                risk_category = "high"
            
            # Calculate feature importance-based risk factors
            factors = calculate_ml_factors(scaled_features[0])
            
            # Return ML result with factors for visualization
            return {
                'prediction': 1 if risk_score > 50 else 0,
                'probabilities': [probabilities[0], probabilities[1]],
                'risk_level': risk_level,
                'risk_category': risk_category,
                'overall_risk': risk_score,
                'factors': factors
            }
        except Exception as e:
            return {
                'error': f"ML prediction failed: {str(e)}"
            }
    else:
        return {
            'error': "ML model or scaler not loaded"
        }

def calculate_ml_factors(scaled_features):
    """Calculate risk factors using ML feature importances"""
    if ml_model and feature_names:
        try:
            # Get feature importances
            importances = ml_model.feature_importances_
            feature_importance_dict = dict(zip(feature_names, importances))
            
            # Normalize to 0-100 scale
            max_importance = max(importances) if len(importances) > 0 else 1
            factors = {}
            
            # Calculate each factor's contribution
            for i, feature in enumerate(feature_names):
                # Scale importance to 0-100 range
                scaled_importance = (feature_importance_dict[feature] / max_importance) * 100
                
                # Adjust based on feature value
                feature_value = scaled_features[i]
                value_contribution = abs(feature_value) * 20  # Scale factor
                
                # Combine importance and value contribution
                factor_score = min(100, scaled_importance + value_contribution)
                
                # Map to factor names used in UI
                factor_name = feature_to_factor(feature)
                factors[factor_name] = factor_score
                
            return factors
        except:
            # Fallback to neutral factors
            return {
                'complexityRisk': 50,
                'overdueRisk': 50,
                'teamRisk': 50,
                'successRisk': 50,
                'timelineRisk': 50,
                'budgetRisk': 50,
                'activityRisk': 50
            }
    else:
        # Return neutral factors if model not available
        return {
            'complexityRisk': 50,
            'overdueRisk': 50,
            'teamRisk': 50,
            'successRisk': 50,
            'timelineRisk': 50,
            'budgetRisk': 50,
            'activityRisk': 50
        }

def feature_to_factor(feature_name):
    """Map ML feature names to UI factor names"""
    mapping = {
        'project_complexity': 'complexityRisk',
        'num_overdue_tasks': 'overdueRisk',
        'num_assignees': 'teamRisk',
        'owner_success_rate': 'successRisk',
        'planned_duration_days': 'timelineRisk',
        'initial_budget_usd': 'budgetRisk',
        'daily_task_updates': 'activityRisk'
    }
    return mapping.get(feature_name, feature_name)

def display_risk_card(risk_level, risk_category, overall_risk):
    """Display styled risk card"""
    risk_emoji = "🟢" if risk_category == 'low' else "🟡" if risk_category == 'medium' else "🔴"
    return f"""
    <div class="risk-card {risk_category}-risk">
        <h2>{risk_emoji} {risk_level}</h2>
        <h3>Risk Score: {overall_risk:.1f}%</h3>
    </div>
    """

# --- Visualization Functions ---
def create_enhanced_risk_analysis(prediction_result):
    fig = go.Figure()
    
    # Risk gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prediction_result['overall_risk'],
        domain={'x': [0.1, 0.9], 'y': [0.7, 1]},
        title={'text': "Overall Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},
                {'range': [30, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction_result['overall_risk']
            }
        }
    ))
    
    # Risk factors bar chart
    factor_names = ['Complexity', 'Overdue', 'Team', 'Success', 'Timeline', 'Budget', 'Activity']
    factor_values = [
        prediction_result['factors']['complexityRisk'],
        prediction_result['factors']['overdueRisk'],
        prediction_result['factors']['teamRisk'],
        prediction_result['factors']['successRisk'],
        prediction_result['factors']['timelineRisk'],
        prediction_result['factors']['budgetRisk'],
        prediction_result['factors']['activityRisk']
    ]
    
    colors = []
    for value in factor_values:
        if value < 30:
            colors.append('#2ecc71')
        elif value < 60:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')
    
    fig.add_trace(go.Bar(
        x=factor_names,
        y=factor_values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in factor_values],
        textposition='auto',
        name="Risk Factors",
        textfont=dict(color='white', size=12)
    ))
    
    # Layout adjustments
    fig.update_layout(
        barmode='group',
        bargap=0.3,
        title={'text': "🔍 Risk Analysis Dashboard", 'x': 0.5},
        height=600,
        showlegend=False,
        yaxis=dict(domain=[0, 0.5]),
        yaxis2=dict(domain=[0.6, 1])
    )
    
    return fig

def create_timeline_projection(input_data):
    total_days = input_data['planned_duration_days']
    days = list(range(0, total_days + 1, max(1, total_days // 20)))
    
    progress = []
    risk_factor = 1 + (input_data['num_overdue_tasks'] * 0.1)
    
    for day in days:
        if day == 0:
            progress.append(0)
        else:
            base_progress = 100 * (1 - np.exp(-3 * day / total_days))
            adjusted_progress = base_progress / risk_factor
            progress.append(min(100, adjusted_progress))
    
    fig = go.Figure()
    
    # Planned progress
    fig.add_trace(go.Scatter(
        x=days,
        y=[100 * day / total_days for day in days],
        mode='lines',
        name='Planned Progress',
        line=dict(color='#2ecc71', width=3, dash='dash')
    ))
    
    # Projected progress
    fig.add_trace(go.Scatter(
        x=days,
        y=progress,
        mode='lines+markers',
        name='Projected Progress',
        line=dict(color='#667eea', width=4),
        marker=dict(size=6, color='#764ba2'),
        fill='tonexty'
    ))
    
    # Milestones
    milestones = [total_days * 0.25, total_days * 0.5, total_days * 0.75, total_days]
    milestone_names = ['25%', '50%', '75%', '100%']
    
    for i, (day, name) in enumerate(zip(milestones, milestone_names)):
        fig.add_vline(
            x=day,
            line_dash="dot",
            line_color="gray",
            annotation_text=name,
            annotation_position="top"
        )
    
    fig.update_layout(
        title={'text': "📅 Project Timeline Projection", 'x': 0.5},
        xaxis_title="Days",
        yaxis_title="Progress (%)",
        height=400
    )
    
    return fig

def create_budget_allocation_chart(input_data, prediction_result):
    """Create pie chart for budget allocation suggestions"""
    total_budget = input_data['initial_budget_usd']
    risk_score = prediction_result['overall_risk']
    
    # Base allocations
    allocations = {
        'Development': 35,
        'Team & Resources': 25,
        'QA & Testing': 15,
        'Project Management': 10,
        'Contingency': 15
    }
    
    # Adjust based on risk factors
    risk_factor = risk_score / 100
    
    # Increase contingency for higher risk
    allocations['Contingency'] = min(30, 15 + (risk_factor * 20))
    
    # Adjust other categories proportionally
    reduction_factor = (allocations['Contingency'] - 15) / 4
    for category in ['Development', 'Team & Resources', 'QA & Testing', 'Project Management']:
        allocations[category] = max(5, allocations[category] - reduction_factor)
    
    # Create labels with percentages and dollar amounts
    labels = []
    values = []
    for category, percent in allocations.items():
        amount = total_budget * percent / 100
        labels.append(f"{category} ({percent:.0f}%)")
        values.append(amount)
    
    # Create pie chart
    fig = px.pie(
        names=labels,
        values=values,
        title='💰 Budget Allocation Recommendation',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})"
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_x=0.5,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    
    return fig

# --- Helper Functions ---
def generate_risk_insights(input_data, prediction_result):
    insights = []
    
    if input_data.get('initial_budget_usd', 0) < 10000:
        insights.append("⚠️ Budget Risk: Low budget may lead to resource constraints")
    elif input_data.get('initial_budget_usd', 0) > 100000:
        insights.append("💰 Budget: Large budget requires careful financial management")
    
    if input_data.get('planned_duration_days', 0) < 30:
        insights.append("⏰ Timeline Risk: Very short timeline increases pressure")
    elif input_data.get('planned_duration_days', 0) > 180:
        insights.append("📅 Timeline: Long duration projects need milestone tracking")
    
    if input_data.get('num_assignees', 0) < 3:
        insights.append("👥 Team Risk: Small team may face capacity issues")
    elif input_data.get('num_assignees', 0) > 15:
        insights.append("🏢 Team: Large team requires strong coordination")
    
    if input_data.get('project_complexity', 0) > 300:
        insights.append(" Complexity Risk: High complexity requires careful planning")
    
    if input_data.get('owner_success_rate', 1) < 0.7:
        insights.append("📊 Performance Risk: Low success rate indicates potential issues")
    
    if input_data.get('daily_task_updates', 0) < 1.0:
        insights.append("📈 Activity Risk: Low daily updates suggest poor tracking")
    
    if input_data.get('num_overdue_tasks', 0) > 10:
        insights.append(" Critical: High number of overdue tasks needs attention")
    
    return insights

def generate_analytics_insights(input_data):
    insights = []
    
    if input_data['project_complexity'] > 300:
        insights.append("📊 High Complexity: Consider breaking down large tasks")
    elif input_data['project_complexity'] > 150:
        insights.append("📊 Moderate Complexity: Maintain clear documentation")
    else:
        insights.append("📊 Low Complexity: Opportunity to deliver quickly")
    
    if input_data['num_overdue_tasks'] > 10:
        insights.append("⏱️ Critical Overdue Tasks: Prioritize immediately")
    elif input_data['num_overdue_tasks'] > 5:
        insights.append("⏱️ Moderate Overdue Tasks: Review dependencies")
    else:
        insights.append("⏱️ Minimal Overdue Tasks: Maintain momentum")
    
    team_size = input_data['num_assignees']
    if team_size < 5:
        insights.append(f"👥 Small Team: Consider adding resources")
    elif team_size > 10:
        insights.append(f"👥 Large Team: Implement clear communication")
    else:
        insights.append(f"👥 Optimal Team Size: Maintain clear roles")
    
    return insights

def get_ai_improvement_suggestions(input_data, prediction_result, insights):
    co = get_cohere_client()
    
    if not co:
        return [
            "Implement regular project status meetings",
            "Create detailed project documentation",
            "Establish clear communication channels",
            "Set up milestone checkpoints",
            "Add buffer time for unexpected challenges"
        ]
    
    try:
        context = f"""
        Project Analysis:
        - Complexity: {input_data['project_complexity']} tasks
        - Budget: ${input_data['initial_budget_usd']:,}
        - Duration: {input_data['planned_duration_days']} days
        - Team Size: {input_data['num_assignees']} people
        - Success Rate: {input_data['owner_success_rate']:.1%}
        - Daily Updates: {input_data['daily_task_updates']:.1f}
        - Overdue Tasks: {input_data['num_overdue_tasks']}
        
        Risk Level: {prediction_result['risk_level']}
        Risk Score: {prediction_result['overall_risk']:.1f}%
        
        Key Insights: {'; '.join(insights)}
        """
        
        prompt = f"""
        As an expert project management consultant, analyze this project data and provide 5 specific, actionable improvement recommendations.
        
        {context}
        
        Please provide practical, implementable suggestions that directly address the identified risks. 
        Each suggestion should be specific, actionable, and focused on measurable outcomes.
        
        Format each suggestion as a clear action item.
        """
        
        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        
        if response and response.generations:
            suggestions_text = response.generations[0].text.strip()
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and len(line) > 20:
                    # Clean up numbering
                    if line[0].isdigit() and '.' in line[:5]:
                        line = line.split('.', 1)[1].strip()
                    if line.startswith('-'):
                        line = line[1:].strip()
                    suggestions.append(line)
            
            return suggestions[:5] if suggestions else [
                "Focus on resolving overdue tasks first",
                "Break down complex tasks into smaller components",
                "Implement daily standups to track progress",
                "Review task dependencies regularly",
                "Allocate contingency time for risks"
            ]
        else:
            return [
                "Communicate frequently with stakeholders",
                "Review resource allocation weekly",
                "Implement automated progress tracking",
                "Conduct risk assessment workshops",
                "Optimize team communication channels"
            ]
    except:
        return [
            "Prioritize critical path tasks",
            "Implement quality assurance checkpoints",
            "Create a risk mitigation plan",
            "Monitor budget utilization weekly",
            "Schedule regular team feedback sessions"
        ]

# --- Generate Sample Data ---
def generate_sample_data():
    np.random.seed(42)
    sample_data = []
    for i in range(10):
        sample = {
            'project_complexity': np.random.randint(100, 300),
            'num_overdue_tasks': np.random.randint(0, 15),
            'num_assignees': np.random.randint(3, 12),
            'owner_success_rate': round(np.random.uniform(0.6, 0.95), 2),
            'planned_duration_days': np.random.randint(60, 180),
            'daily_task_updates': round(np.random.uniform(1.0, 4.0), 1),
            'initial_budget_usd': np.random.randint(25000, 100000)
        }
        sample_data.append(sample)
    return pd.DataFrame(sample_data)

# --- Main App ---
def main():
    # Initialize session state
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Project Risk Predictor</h1>
        <p>Intelligent risk assessment powered by machine learning and AI insights</p>
        <p><small>Advanced analytics • Real-time predictions • AI-powered recommendations</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show service status
    col1, col2 = st.columns(2)
    with col1:
        co_client = get_cohere_client()
        if co_client:
            st.success("✅ Cohere AI Connected - Intelligent suggestions enabled")
        else:
            st.warning("⚠️ Cohere AI Not Available - Using standard suggestions")
    
    with col2:
        if ml_model:
            st.success("✅ ML Predictive Model Active")
        else:
            st.error("❌ ML Model Not Available - System offline")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        st.markdown("""
        **🎯 Features:**
        - 📝 Manual project input
        - 📁 CSV file upload
        - 📊 Enhanced analytics
        - 🎯 Risk prediction
        - 🤖 AI recommendations
        """)
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This app predicts project risks based on:
        - Project complexity
        - Team dynamics
        - Timeline constraints
        - Budget considerations
        - Activity patterns
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📁 File Upload", "📊 Analysis Dashboard"])
    
    with tab1:
        st.header("📝 Project Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Project Details")
            project_complexity = st.slider("Project Complexity (# Tasks)", 50, 500, 150, 10)
            num_overdue_tasks = st.slider("Number of Overdue Tasks", 0, 100, 5, 1)
            num_assignees = st.slider("Number of Team Members", 1, 20, 5, 1)
            owner_success_rate = st.slider("Owner Success Rate", 0.0, 1.0, 0.85, 0.01, format="%.2f")
        
        with col2:
            st.subheader("⏱️ Timeline & Resources")
            planned_duration_days = st.slider("Planned Duration (Days)", 30, 365, 90, 5)
            daily_task_updates = st.slider("Daily Task Updates", 0.0, 10.0, 2.0, 0.1)
            initial_budget_usd = st.slider("Initial Budget (USD)", 5000, 200000, 50000, 5000, format="$%d")
        
        input_data = {
            'project_complexity': project_complexity,
            'num_overdue_tasks': num_overdue_tasks,
            'num_assignees': num_assignees,
            'owner_success_rate': owner_success_rate,
            'planned_duration_days': planned_duration_days,
            'daily_task_updates': daily_task_updates,
            'initial_budget_usd': initial_budget_usd
        }
        
        # Real-time risk preview
        if ml_model:
            preview_result = predict_project_risk(input_data)
            if 'error' in preview_result:
                st.error(f"⚠️ Prediction error: {preview_result['error']}")
            else:
                st.markdown(display_risk_card(
                    preview_result['risk_level'],
                    preview_result['risk_category'],
                    preview_result['overall_risk']
                ), unsafe_allow_html=True)
        else:
            st.error("❌ Model not loaded - predictions unavailable")
        
        if ml_model and st.button("🔍 Analyze Project Risk", type="primary", use_container_width=True):
            st.session_state.analysis_data = input_data
            st.session_state.analysis_source = "manual"
    
    with tab2:
        st.header("📁 Upload Project Data")
        
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload Your Project Data</h3>
            <p>Upload a CSV file to analyze multiple projects</p>
            <p><small>Supports batch analysis of multiple projects</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show required format
        with st.expander("📋 View Required CSV Format", expanded=False):
            st.markdown("""
            **Required columns:**
            - `project_complexity`: Number of tasks (e.g., 150)
            - `num_overdue_tasks`: Number of overdue tasks (e.g., 5)
            - `num_assignees`: Number of team members (e.g., 5)
            - `owner_success_rate`: Success rate 0.0-1.0 (e.g., 0.85)
            - `planned_duration_days`: Project duration in days (e.g., 90)
            - `daily_task_updates`: Average daily updates (e.g., 2.0)
            - `initial_budget_usd`: Project budget in USD (e.g., 50000)
            """)
            
            # Show sample data
            sample_df = generate_sample_data()
            st.markdown("**Sample data format:**")
            st.dataframe(sample_df.head(3), use_container_width=True)
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help="Upload a CSV file with project data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(df)} projects.")
                
                # Display data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Check for required columns
                required_columns = ['project_complexity', 'num_overdue_tasks', 'num_assignees', 
                                  'owner_success_rate', 'planned_duration_days', 'daily_task_updates', 
                                  'initial_budget_usd']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                else:
                    st.success("✅ All required columns found!")
                    
                    # Select project to analyze
                    if len(df) > 1:
                        project_index = st.selectbox(
                            "Select project to analyze:",
                            range(len(df)),
                            format_func=lambda x: f"Project {x+1}"
                        )
                    else:
                        project_index = 0
                    
                    selected_project = df.iloc[project_index]
                    input_data = selected_project[required_columns].to_dict()
                    
                    # Display selected project
                    st.subheader(f"📋 Selected Project (Row {project_index + 1})")
                    
                    # Display risk card
                    if ml_model:
                        preview_result = predict_project_risk(input_data)
                        if 'error' in preview_result:
                            st.error(f"⚠️ Prediction error: {preview_result['error']}")
                        else:
                            st.markdown(display_risk_card(
                                preview_result['risk_level'],
                                preview_result['risk_category'],
                                preview_result['overall_risk']
                            ), unsafe_allow_html=True)
                    else:
                        st.error("❌ Model not loaded - predictions unavailable")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Complexity", f"{input_data['project_complexity']:.0f} tasks")
                        st.metric("Overdue Tasks", f"{input_data['num_overdue_tasks']:.0f}")
                    
                    with col2:
                        st.metric("Team Size", f"{input_data['num_assignees']:.0f} people")
                        st.metric("Success Rate", f"{input_data['owner_success_rate']:.1%}")
                    
                    with col3:
                        st.metric("Duration", f"{input_data['planned_duration_days']:.0f} days")
                        st.metric("Budget", f"${input_data['initial_budget_usd']:,.0f}")
                    
                    if ml_model and st.button("🔍 Analyze Selected Project", type="primary"):
                        st.session_state.analysis_data = input_data
                        st.session_state.analysis_source = "file"
                    elif not ml_model:
                        st.error("❌ Model not loaded - predictions unavailable")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        else:
            # Download sample file
            sample_df = generate_sample_data()
            csv = sample_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=csv,
                file_name="sample_project_data.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("📊 Analysis Dashboard")
        
        if st.session_state.get('analysis_data'):
            input_data = st.session_state.analysis_data
            source = st.session_state.get('analysis_source', 'manual').capitalize()
            
            st.success(f"Showing analysis from {source} Input")
            
            if ml_model:
                # Make prediction
                prediction_result = predict_project_risk(input_data)
                
                if 'error' in prediction_result:
                    st.error(f"❌ Prediction failed: {prediction_result['error']}")
                else:
                    # Determine risk level
                    risk_level = prediction_result['risk_level']
                    risk_category = prediction_result['risk_category']
                    risk_color = f"{risk_category}-risk"
                    
                    # Display results
                    st.markdown(display_risk_card(
                        risk_level,
                        risk_category,
                        prediction_result['overall_risk']
                    ), unsafe_allow_html=True)
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Complexity", f"{input_data['project_complexity']} tasks")
                    col2.metric("Team Size", f"{input_data['num_assignees']} people")
                    col3.metric("Budget", f"${input_data['initial_budget_usd']:,.0f}")
                    
                    # Risk analysis visualization
                    st.subheader("🔍 Risk Analysis")
                    fig_risk = create_enhanced_risk_analysis(prediction_result)
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Timeline projection
                    st.subheader("📅 Timeline Projection")
                    fig_timeline = create_timeline_projection(input_data)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Budget allocation chart
                    st.subheader("💰 Budget Allocation Recommendation")
                    budget_chart = create_budget_allocation_chart(input_data, prediction_result)
                    st.plotly_chart(budget_chart, use_container_width=True)
                    
                    # Insights
                    st.subheader("📋 Key Insights")
                    insights = generate_risk_insights(input_data, prediction_result)
                    for insight in insights:
                        st.markdown(f"<div class='metric-card'>{insight}</div>", unsafe_allow_html=True)
                    
                    # AI-powered recommendations
                    st.subheader("🤖 AI-Powered Recommendations")
                    with st.spinner("Generating intelligent recommendations..."):
                        suggestions = get_ai_improvement_suggestions(input_data, prediction_result, insights)
                        for i, suggestion in enumerate(suggestions, 1):
                            st.markdown(f"""
                            <div class="improvement-suggestion">
                                <h4>Recommendation {i}</h4>
                                <p>{suggestion}</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("❌ Model not loaded - predictions unavailable")
        else:
            st.info("Please analyze a project first in the Manual Input or File Upload tab")
            st.markdown("""
            **How to get started:**
            1. Go to the **Manual Input** tab to enter project details
            2. Or go to the **File Upload** tab to analyze from a CSV file
            3. Click "Analyze Project Risk" to see results here
            """)

if __name__ == "__main__":
    main()