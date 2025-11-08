import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Shooting Score Predictor",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .session-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #e6e6e6;
        margin: 10px 0;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class ShootingScorePredictor:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def calculate_session_total(self, series_scores):
        """Calculate total score for a session from 6 series"""
        return sum(series_scores)
    
    def prepare_features(self, session_totals):
        """Prepare features from last 3 session totals"""
        features = []
        # Last 3 session totals
        features.extend(session_totals)
        # Average of last 3 sessions
        features.append(np.mean(session_totals))
        # Trend (slope of last 3 sessions)
        x = np.array([1, 2, 3]).reshape(-1, 1)
        y = np.array(session_totals)
        if len(y) == 3:
            trend = LinearRegression().fit(x, y).coef_[0]
        else:
            trend = 0
        features.append(trend)
        # Standard deviation (consistency across sessions)
        features.append(np.std(session_totals))
        # Improvement rate
        features.append(session_totals[2] - session_totals[0])
        # Recent momentum
        features.append(session_totals[2] - session_totals[1])
        
        return np.array(features).reshape(1, -1)
    
    def predict_next_session(self, sessions_data, model_name='Linear Regression'):
        """Predict next session total based on last 3 sessions with realistic fluctuations"""
        if len(sessions_data) != 3:
            return None, "Please enter exactly 3 sessions", []
        
        # Calculate session totals and series statistics
        session_totals = [self.calculate_session_total(session) for session in sessions_data]
        all_series_scores = [score for session in sessions_data for score in session]
        
        # Calculate consistency metrics
        session_consistency = np.std(session_totals)
        series_consistency = np.std(all_series_scores)
        
        # Use simple prediction if scikit-learn fails
        try:
            model = self.models[model_name]
            
            # Generate realistic training data with various patterns
            X_train = []
            y_train = []
            
            # Base patterns from actual shooting data with realistic fluctuations
            base_totals = [600, 610, 615, 620, 625, 630, 635, 640, 645, 650]
            
            for base in base_totals:
                # Consistent performance patterns (small fluctuations)
                X_train.append(self.prepare_features([base-1, base, base+1])[0])
                y_train.append(base + np.random.normal(0, 0.5))  # Small random fluctuation
                
                X_train.append(self.prepare_features([base, base, base])[0])
                y_train.append(base + np.random.normal(0, 0.3))  # Very stable
                
                # Improving patterns
                X_train.append(self.prepare_features([base-3, base-1, base+1])[0])
                y_train.append(base + 2 + np.random.normal(0, 1))
                
                # Declining patterns
                X_train.append(self.prepare_features([base+2, base, base-1])[0])
                y_train.append(base - 2 + np.random.normal(0, 1))
                
                # Fluctuating patterns
                X_train.append(self.prepare_features([base+2, base-1, base+1])[0])
                y_train.append(base + np.random.normal(0, 2))
                
                # Plateau patterns
                X_train.append(self.prepare_features([base+1, base+1, base])[0])
                y_train.append(base + np.random.normal(0, 0.5))
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make prediction
            features = self.prepare_features(session_totals)
            prediction = model.predict(features)[0]
            
            # Add realistic noise based on shooter's consistency
            if series_consistency < 1.0:  # Very consistent shooter
                prediction_noise = np.random.normal(0, 0.5)
            elif series_consistency < 2.0:  # Consistent shooter
                prediction_noise = np.random.normal(0, 1.0)
            else:  # Inconsistent shooter
                prediction_noise = np.random.normal(0, 2.0)
                
            prediction += prediction_noise
            
        except Exception as e:
            # Fallback to realistic prediction based on patterns
            avg_score = np.mean(session_totals)
            trend = session_totals[2] - session_totals[0]
            recent_momentum = session_totals[2] - session_totals[1]
            
            # Realistic prediction considering various scenarios
            if session_consistency < 2.0:  # Very consistent
                # Small improvement or maintenance
                if trend > 1:
                    prediction = avg_score + min(trend * 0.3, 2)
                elif trend < -1:
                    prediction = avg_score + max(trend * 0.3, -2)
                else:
                    prediction = avg_score + np.random.normal(0, 0.5)
            else:  # Inconsistent
                # Larger fluctuations possible
                prediction = avg_score + np.random.normal(0, min(session_consistency * 0.5, 5))
        
        # Calculate confidence based on consistency
        confidence_range = max(2.0, session_consistency * 1.2)
        
        # Ensure prediction is realistic
        max_possible = 659.4
        prediction = max(500, min(prediction, max_possible))
        
        return prediction, confidence_range, session_totals

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Shooting Score Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict next session performance based on last 3 sessions (6 series each)")
    
    # Initialize predictor
    predictor = ShootingScorePredictor()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    athlete_name = st.sidebar.text_input("Athlete Name", "Tom Cruise")
    category = st.sidebar.selectbox("Category", [
        "Sub Youth Men","Sub Youth Women", "Youth Men", "Youth Women" ,"Junior Men" ,"Junior Women","Junior Men"
    ])
    
    model_choice = st.sidebar.selectbox(
        "Prediction Model",
        ["Linear Regression", "Random Forest"]
    )
    
    # Main content - Session Input Section
    st.markdown("### 📊 Enter Last 3 Sessions (6 Series Each)")
    
    sessions_data = []
    
    # Session 1
    with st.container():
        st.markdown('<div class="session-box">', unsafe_allow_html=True)
        st.subheader("Session 1 (Oldest)")
        
        cols = st.columns(6)
        session1_series = []
        for i, col in enumerate(cols):
            with col:
                score = col.number_input(
                    f"S{i+1}", 
                    min_value=0.0, 
                    max_value=109.9, 
                    value=[102.2, 104.1, 105.3, 104.1, 103.3, 104.1][i],
                    step=0.1,
                    key=f"s1_{i}"
                )
                session1_series.append(score)
                st.caption(f"Series {i+1}")
        
        session1_total = sum(session1_series)
        st.metric("Session 1 Total", f"{session1_total:.1f}")
        sessions_data.append(session1_series)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Session 2
    with st.container():
        st.markdown('<div class="session-box">', unsafe_allow_html=True)
        st.subheader("Session 2")
        
        cols = st.columns(6)
        session2_series = []
        for i, col in enumerate(cols):
            with col:
                score = col.number_input(
                    f"S{i+1}", 
                    min_value=0.0, 
                    max_value=109.9, 
                    value=[103.5, 104.2, 104.8, 103.6, 102.7, 103.2][i],
                    step=0.1,
                    key=f"s2_{i}"
                )
                session2_series.append(score)
                st.caption(f"Series {i+1}")
        
        session2_total = sum(session2_series)
        st.metric("Session 2 Total", f"{session2_total:.1f}")
        sessions_data.append(session2_series)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Session 3 (Most Recent)
    with st.container():
        st.markdown('<div class="session-box">', unsafe_allow_html=True)
        st.subheader("Session 3 (Most Recent)")
        
        cols = st.columns(6)
        session3_series = []
        for i, col in enumerate(cols):
            with col:
                score = col.number_input(
                    f"S{i+1}", 
                    min_value=0.0, 
                    max_value=109.9, 
                    value=[104.4, 103.5, 104.7, 104.7, 105.1, 104.0][i],
                    step=0.1,
                    key=f"s3_{i}"
                )
                session3_series.append(score)
                st.caption(f"Series {i+1}")
        
        session3_total = sum(session3_series)
        st.metric("Session 3 Total", f"{session3_total:.1f}")
        sessions_data.append(session3_series)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance Analysis
    st.markdown("### 📈 Performance Analysis")
    
    if len(sessions_data) == 3:
        session_totals = [predictor.calculate_session_total(session) for session in sessions_data]
        all_series = [score for session in sessions_data for score in session]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            avg_session = np.mean(session_totals)
            st.metric("Average Session", f"{avg_session:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            trend = session_totals[2] - session_totals[0]
            st.metric("3-Session Trend", f"{trend:+.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            session_consistency = np.std(session_totals)
            st.metric("Session Consistency", f"{session_consistency:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            series_consistency = np.std(all_series)
            st.metric("Series Consistency", f"{series_consistency:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance Pattern Analysis
        st.markdown("#### 🎯 Performance Pattern")
        if session_consistency < 2.0:
            st.success("**Consistent Performer**: Stable performance with minimal fluctuations")
        elif session_consistency < 4.0:
            st.info("**Balanced Performer**: Moderate consistency with some variability")
        else:
            st.warning("**Variable Performer**: Significant session-to-session fluctuations")
        
        # Series-wise analysis
        st.markdown("#### 📊 Series-wise Performance")
        series_avgs = np.array(sessions_data).mean(axis=0)
        
        cols = st.columns(6)
        for i, col in enumerate(cols):
            with col:
                series_scores = [session[i] for session in sessions_data]
                avg_series = np.mean(series_scores)
                trend_series = series_scores[2] - series_scores[0]
                consistency_series = np.std(series_scores)
                
                st.metric(
                    f"Series {i+1}", 
                    f"{avg_series:.1f}",
                    delta=f"{trend_series:+.1f}"
                )
                st.caption(f"σ: {consistency_series:.2f}")
    
    # Prediction Section
    st.markdown("### 🔮 Next Session Prediction")
    
    if st.button("🎯 Predict Next Session", type="primary"):
        prediction, confidence_range, session_totals = predictor.predict_next_session(sessions_data, model_choice)
        
        if prediction is not None:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                current_session = session_totals[2]
                improvement = prediction - current_session
                
                st.metric(
                    "Predicted Session Total", 
                    f"{prediction:.1f}",
                    delta=f"{improvement:+.1f} from last session"
                )
                
                # Progress visualization
                max_possible = 659.4
                progress_percent = min(100, (prediction / max_possible) * 100)
                st.progress(int(progress_percent))
                
                # Expected range
                lower_bound = max(500, prediction - confidence_range)
                upper_bound = min(max_possible, prediction + confidence_range)
                st.write(f"**Expected Range:** {lower_bound:.1f} - {upper_bound:.1f}")
                st.write(f"**Model:** {model_choice}")
                
            with col2:
                # Performance indicator
                if improvement > 3:
                    st.success("🚀 Strong Improvement")
                elif improvement > 1:
                    st.info("📈 Mild Improvement")
                elif improvement > -1:
                    st.info("➡️ Stable Performance")
                elif improvement > -3:
                    st.warning("📉 Mild Decline")
                else:
                    st.error("📉 Significant Decline")
                
                # Confidence indicator based on consistency
                session_consistency = np.std(session_totals)
                if session_consistency < 2:
                    st.success("High Confidence")
                elif session_consistency < 4:
                    st.warning("Medium Confidence")
                else:
                    st.error("Low Confidence")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Training Recommendations
            st.markdown("### 💡 Training Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.write("**Session Strategy:**")
                session_consistency = np.std(session_totals)
                trend = session_totals[2] - session_totals[0]
                
                if session_consistency > 4:
                    st.write("• 🎯 Focus on session-to-session consistency")
                    st.write("• 📊 Identify causes of performance fluctuations")
                elif session_consistency > 2:
                    st.write("• ⚖️ Work on maintaining stable performance")
                else:
                    st.write("• ✅ Excellent consistency - maintain focus")
                
                if trend > 3:
                    st.write("• 🔥 Capitalize on positive momentum")
                elif trend < -2:
                    st.write("• 📉 Address performance decline")
                
                st.write("• 🧠 Mental preparation between sessions")
            
            with rec_col2:
                st.write("**Series-specific Focus:**")
                # Identify weakest and strongest series
                series_avgs = np.array(sessions_data).mean(axis=0)
                series_consistencies = [np.std([session[i] for session in sessions_data]) for i in range(6)]
                
                weakest_series = np.argmin(series_avgs) + 1
                strongest_series = np.argmax(series_avgs) + 1
                most_variable = np.argmax(series_consistencies) + 1
                
                st.write(f"• 💪 Strengthen Series {weakest_series} (lowest avg)")
                st.write(f"• ✅ Maintain Series {strongest_series} (highest avg)")
                st.write(f"• 🎯 Stabilize Series {most_variable} (most variable)")
                st.write("• ⚡ Build endurance for consistent performance")
        
        else:
            st.error("Please complete all 3 sessions with 6 series each")

    # Example section
    with st.expander("💡 How to use this predictor"):
        st.write("""
        **Realistic Performance Patterns:**
        - **Consistent Shooters**: Small fluctuations (±1-2 points)
        - **Improving Shooters**: Gradual improvements (1-3 points per session)
        - **Declining Shooters**: Performance drops due to various factors
        - **Variable Shooters**: Larger fluctuations based on consistency
        
        **Key Metrics:**
        - **Session Consistency**: How stable are your session totals
        - **Series Consistency**: How consistent are your individual series
        - **Trend Direction**: Are you improving, declining, or stable
        
        **Prediction Factors:**
        - Based on your historical consistency
        - Considers both improvement and decline scenarios
        - Accounts for realistic performance fluctuations
        - Adapts to your shooting pattern
        """)

if __name__ == "__main__":
    main()