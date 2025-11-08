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
    .series-row {
        display: flex;
        justify-content: space-between;
        margin: 5px 0;
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
        trend = LinearRegression().fit(x, y).coef_[0]
        features.append(trend)
        # Standard deviation (consistency across sessions)
        features.append(np.std(session_totals))
        # Improvement rate
        features.append(session_totals[2] - session_totals[0])
        
        return np.array(features).reshape(1, -1)
    
    def predict_next_session(self, sessions_data, model_name='Linear Regression'):
        """Predict next session total based on last 3 sessions"""
        if len(sessions_data) != 3:
            return None, "Please enter exactly 3 sessions"
        
        # Calculate session totals
        session_totals = [self.calculate_session_total(session) for session in sessions_data]
        
        # Train model
        model = self.models[model_name]
        
        # Generate training data based on shooting patterns
        X_train = []
        y_train = []
        
        # Base patterns from actual shooting data
        base_totals = [610, 615, 620, 625, 630]
        for base in base_totals:
            # Consistent improvement patterns
            X_train.append(self.prepare_features([base-2, base, base+1])[0])
            y_train.append(base + 2)
            
            X_train.append(self.prepare_features([base, base+1, base+2])[0])
            y_train.append(base + 3)
            
            # Stable performance patterns
            X_train.append(self.prepare_features([base, base, base])[0])
            y_train.append(base)
            
            # Fluctuating patterns
            X_train.append(self.prepare_features([base-1, base+2, base])[0])
            y_train.append(base + 1)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make prediction
        features = self.prepare_features(session_totals)
        prediction = model.predict(features)[0]
        
        # Calculate confidence based on consistency
        consistency = np.std(session_totals)
        confidence_range = max(2.0, consistency * 1.5)
        
        return prediction, confidence_range, session_totals

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Shooting Score Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict next session performance based on last 3 sessions (6 series each)")
    
    # Initialize predictor
    predictor = ShootingScorePredictor()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    athlete_name = st.sidebar.text_input("Athlete Name", "Suryansh Narayan")
    category = st.sidebar.selectbox("Category", [
        "Junior Men", "Youth Men", "Youth Women", "Sub Youth Men", "Sub Youth Women"
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
                    max_value=105.0, 
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
                    max_value=105.0, 
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
                    max_value=105.0, 
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
            consistency = np.std(session_totals)
            st.metric("Session Consistency", f"{consistency:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            recent_improvement = session_totals[2] - session_totals[1]
            st.metric("Recent Change", f"{recent_improvement:+.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Series-wise analysis
        st.markdown("#### 📊 Series-wise Performance")
        series_avgs = np.array(sessions_data).mean(axis=0)
        
        cols = st.columns(6)
        for i, col in enumerate(cols):
            with col:
                series_scores = [session[i] for session in sessions_data]
                avg_series = np.mean(series_scores)
                trend_series = series_scores[2] - series_scores[0]
                
                st.metric(
                    f"Series {i+1} Avg", 
                    f"{avg_series:.1f}",
                    delta=f"{trend_series:+.1f}"
                )
    
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
                max_possible = 630  # 6 series × 105 max each
                progress_percent = min(100, (prediction / max_possible) * 100)
                st.progress(int(progress_percent))
                
                # Expected range
                lower_bound = max(0, prediction - confidence_range)
                upper_bound = min(630, prediction + confidence_range)
                st.write(f"**Expected Range:** {lower_bound:.1f} - {upper_bound:.1f}")
                st.write(f"**Model:** {model_choice}")
                
            with col2:
                # Performance indicator
                if improvement > 2:
                    st.success("🚀 Strong Improvement")
                elif improvement > 0:
                    st.info("📈 Mild Improvement")
                elif improvement == 0:
                    st.info("➡️ Stable Performance")
                else:
                    st.warning("📉 Slight Dip")
                
                # Confidence indicator
                if confidence_range < 3:
                    st.success("High Confidence")
                elif confidence_range < 5:
                    st.warning("Medium Confidence")
                else:
                    st.error("Variable Pattern")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Training Recommendations
            st.markdown("### 💡 Training Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.write("**Session Strategy:**")
                if consistency > 5:
                    st.write("• 🎯 Focus on session consistency")
                    st.write("• 📊 Analyze performance fluctuations")
                else:
                    st.write("• ✅ Excellent session consistency")
                
                if trend > 3:
                    st.write("• 🔥 Maintain positive momentum")
                elif trend < 0:
                    st.write("• 📉 Address declining trend")
                
                st.write("• 🧠 Mental preparation between sessions")
            
            with rec_col2:
                st.write("**Series-specific Focus:**")
                # Identify weakest series
                series_avgs = np.array(sessions_data).mean(axis=0)
                weakest_series = np.argmin(series_avgs) + 1
                strongest_series = np.argmax(series_avgs) + 1
                
                st.write(f"• 💪 Strengthen Series {weakest_series} (weakest)")
                st.write(f"• ✅ Maintain Series {strongest_series} (strongest)")
                st.write("• ⚡ Improve endurance in later series")
                st.write("• 🎯 Focus on precision in early series")
        
        else:
            st.error("Please complete all 3 sessions with 6 series each")

    # Example section
    with st.expander("💡 How to use this predictor"):
        st.write("""
        **Input Format:**
        - Enter 3 complete sessions (most recent last)
        - Each session has 6 series scores
        - Series scores typically range from 95-105
        
        **Calculation:**
        - Session Total = Sum of all 6 series scores
        - Prediction based on session totals and trends
        
        **Example Session:**
        - Series 1: 102.2
        - Series 2: 104.1  
        - Series 3: 105.3
        - Series 4: 104.1
        - Series 5: 103.3
        - Series 6: 104.1
        - **Total: 623.1**
        """)

if __name__ == "__main__":
    main()