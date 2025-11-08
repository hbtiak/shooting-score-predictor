import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    .analysis-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .session-details {
        background-color: #e8f4fd;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .session-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 3px;
    }
    .trend-positive {
        color: #28a745;
        font-weight: bold;
    }
    .trend-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .trend-neutral {
        color: #6c757d;
        font-weight: bold;
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
    
    def analyze_series_trends(self, sessions_data):
        """Analyze performance trends across series"""
        if len(sessions_data) != 3:
            return None
        
        analysis = {}
        
        # Convert to numpy array for easier analysis
        sessions_array = np.array(sessions_data)
        
        # Calculate average performance per series across all sessions
        series_avgs = sessions_array.mean(axis=0)
        analysis['series_averages'] = series_avgs
        
        # Calculate series trends (improvement/decline within sessions)
        series_trends = []
        for session in sessions_array:
            x = np.array(range(1, 7))
            y = session
            trend = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
            series_trends.append(trend)
        analysis['session_series_trends'] = series_trends
        
        # Calculate consistency per series
        series_consistency = sessions_array.std(axis=0)
        analysis['series_consistency'] = series_consistency
        
        # Calculate performance change from first to last series in each session
        series_delta = []
        for session in sessions_array:
            delta = session[-1] - session[0]  # Last series - first series
            series_delta.append(delta)
        analysis['series_delta'] = series_delta
        
        # Overall series pattern (which series are strongest/weakest)
        analysis['strongest_series'] = np.argmax(series_avgs) + 1
        analysis['weakest_series'] = np.argmin(series_avgs) + 1
        analysis['most_consistent_series'] = np.argmin(series_consistency) + 1
        analysis['least_consistent_series'] = np.argmax(series_consistency) + 1
        
        # Endurance analysis (performance in later series)
        early_series_avg = np.mean(sessions_array[:, :3])  # Series 1-3
        late_series_avg = np.mean(sessions_array[:, 3:])   # Series 4-6
        analysis['endurance_factor'] = late_series_avg - early_series_avg
        
        # Series progression analysis (session-to-session improvement per series)
        series_progression = []
        for series_idx in range(6):
            series_scores = sessions_array[:, series_idx]
            trend = LinearRegression().fit(np.array([1, 2, 3]).reshape(-1, 1), series_scores).coef_[0]
            improvement = series_scores[2] - series_scores[0]
            series_progression.append({
                'trend': trend,
                'improvement': improvement,
                'current_score': series_scores[2],
                'starting_score': series_scores[0]
            })
        analysis['series_progression'] = series_progression
        
        # Identify improving and declining series
        improving_series = []
        declining_series = []
        stable_series = []
        
        for i, prog in enumerate(series_progression):
            if prog['improvement'] > 0.5:
                improving_series.append(i + 1)
            elif prog['improvement'] < -0.5:
                declining_series.append(i + 1)
            else:
                stable_series.append(i + 1)
        
        analysis['improving_series'] = improving_series
        analysis['declining_series'] = declining_series
        analysis['stable_series'] = stable_series
        
        return analysis
    
    def create_series_trend_plot(self, sessions_data, analysis):
        """Create visualization of series performance trends"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Series Performance Across Sessions
        sessions_array = np.array(sessions_data)
        for i, session in enumerate(sessions_array):
            ax1.plot(range(1, 7), session, marker='o', linewidth=2, 
                    label=f'Session {i+1}', alpha=0.8)
        
        ax1.set_xlabel('Series Number')
        ax1.set_ylabel('Score')
        ax1.set_title('Series Performance Across All Sessions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, 7))
        
        # Plot 2: Average Performance per Series
        series_avgs = analysis['series_averages']
        series_std = analysis['series_consistency']
        bars = ax2.bar(range(1, 7), series_avgs, yerr=series_std, 
                      capsize=5, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Series Number')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Performance per Series (± consistency)')
        ax2.set_xticks(range(1, 7))
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, avg in zip(bars, series_avgs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{avg:.1f}', ha='center', va='bottom')
        
        # Plot 3: Performance Trend Within Sessions
        trends = analysis['session_series_trends']
        colors = ['green' if trend > 0 else 'red' for trend in trends]
        bars = ax3.bar(range(1, 4), trends, color=colors, alpha=0.7)
        ax3.set_xlabel('Session')
        ax3.set_ylabel('Trend (points per series)')
        ax3.set_title('Performance Trend Within Each Session\n(Green=Improving, Red=Declining)')
        ax3.set_xticks(range(1, 4))
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, trend in zip(bars, trends):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.15),
                    f'{trend:+.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 4: Endurance Analysis (First vs Last Series)
        first_series_avg = np.mean(sessions_array[:, 0])
        last_series_avg = np.mean(sessions_array[:, -1])
        endurance_data = [first_series_avg, last_series_avg]
        colors = ['lightcoral', 'lightgreen']
        bars = ax4.bar(['First Series', 'Last Series'], endurance_data, 
                      color=colors, alpha=0.7)
        ax4.set_ylabel('Average Score')
        ax4.set_title('Endurance Analysis: First vs Last Series')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, endurance_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_series_progression_plot(self, sessions_data, analysis):
        """Create detailed series progression analysis plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        sessions_array = np.array(sessions_data)
        series_progression = analysis['series_progression']
        
        # Plot 1: Series Progression Over Sessions
        for series_idx in range(6):
            series_scores = sessions_array[:, series_idx]
            ax1.plot([1, 2, 3], series_scores, marker='o', linewidth=2, 
                    label=f'Series {series_idx+1}', alpha=0.8)
        
        ax1.set_xlabel('Session Number')
        ax1.set_ylabel('Score')
        ax1.set_title('Series Progression Over Sessions')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([1, 2, 3])
        
        # Plot 2: Series Improvement Analysis
        improvements = [prog['improvement'] for prog in series_progression]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(range(1, 7), improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Series Number')
        ax2.set_ylabel('Improvement (Session 3 - Session 1)')
        ax2.set_title('Series Improvement: Session 1 to Session 3\n(Green=Improved, Red=Declined)')
        ax2.set_xticks(range(1, 7))
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.1),
                    f'{imp:+.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 3: Current Series Performance (Session 3)
        current_scores = sessions_array[2]  # Most recent session
        avg_score = np.mean(current_scores)
        colors = ['green' if score > avg_score else 'red' for score in current_scores]
        bars = ax3.bar(range(1, 7), current_scores, color=colors, alpha=0.7)
        ax3.axhline(y=avg_score, color='blue', linestyle='--', alpha=0.7, label=f'Average: {avg_score:.1f}')
        ax3.set_xlabel('Series Number')
        ax3.set_ylabel('Score (Session 3)')
        ax3.set_title('Current Series Performance vs Average\n(Green=Above Avg, Red=Below Avg)')
        ax3.set_xticks(range(1, 7))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, current_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Plot 4: Consistency Heatmap
        consistency_data = sessions_array.T  # Transpose to get series as rows
        im = ax4.imshow(consistency_data, cmap='YlOrRd', aspect='auto', 
                       extent=[0.5, 3.5, 6.5, 0.5])
        ax4.set_xlabel('Session')
        ax4.set_ylabel('Series')
        ax4.set_title('Performance Heatmap (Darker = Higher Score)')
        ax4.set_xticks([1, 2, 3])
        ax4.set_yticks(range(1, 7))
        
        # Add value annotations in heatmap
        for i in range(6):  # Series
            for j in range(3):  # Sessions
                ax4.text(j + 1, i + 1, f'{consistency_data[i, j]:.1f}', 
                        ha='center', va='center', color='white' if consistency_data[i, j] > 104 else 'black',
                        fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Score')
        plt.tight_layout()
        return fig
    
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
    session_totals = []
    
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
        session_totals.append(session1_total)
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
        session_totals.append(session2_total)
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
        session_totals.append(session3_total)
        sessions_data.append(session3_series)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Session Details Summary - Updated Compact Format
    if len(sessions_data) == 3:
        st.markdown("### 📋 Session Summary")
        
        # Display sessions in vertical compact format
        session_names = ["Session 1 (Oldest)", "Session 2", "Session 3 (Most Recent)"]
        for i, (session, total) in enumerate(zip(sessions_data, session_totals)):
            st.markdown('<div class="session-details">', unsafe_allow_html=True)
            st.markdown(f'<div class="session-header">{session_names[i]} - Total Score: {total:.1f}</div>', unsafe_allow_html=True)
            st.markdown(f"**Series:** {', '.join([f'{s:.1f}' for s in session])}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Session Progress
        st.markdown("#### 📈 Session Progress")
        progress_col1, progress_col2, progress_col3 = st.columns(3)
        
        with progress_col1:
            if session_totals[1] > session_totals[0]:
                delta1 = f"+{session_totals[1] - session_totals[0]:.1f}"
            else:
                delta1 = f"{session_totals[1] - session_totals[0]:.1f}"
            st.metric("Session 1 → Session 2", f"{session_totals[1]:.1f}", delta=delta1)
        
        with progress_col2:
            if session_totals[2] > session_totals[1]:
                delta2 = f"+{session_totals[2] - session_totals[1]:.1f}"
            else:
                delta2 = f"{session_totals[2] - session_totals[1]:.1f}"
            st.metric("Session 2 → Session 3", f"{session_totals[2]:.1f}", delta=delta2)
        
        with progress_col3:
            if session_totals[2] > session_totals[0]:
                delta3 = f"+{session_totals[2] - session_totals[0]:.1f}"
            else:
                delta3 = f"{session_totals[2] - session_totals[0]:.1f}"
            st.metric("Overall Trend", f"{session_totals[2]:.1f}", delta=delta3)
    
    # Performance Analysis
    st.markdown("### 📈 Performance Analysis")
    
    if len(sessions_data) == 3:
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
    
    # Enhanced Series Trend Analysis
    st.markdown("### 📊 Detailed Series Trend Analysis")
    
    if len(sessions_data) == 3:
        # Perform series analysis
        series_analysis = predictor.analyze_series_trends(sessions_data)
        
        if series_analysis:
            # Display key insights in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                endurance = series_analysis['endurance_factor']
                st.metric("Endurance Factor", f"{endurance:+.2f}")
                st.caption("Late vs Early Series")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                avg_trend = np.mean(series_analysis['session_series_trends'])
                st.metric("Avg Series Trend", f"{avg_trend:+.3f}")
                st.caption("Points per series")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                strongest = series_analysis['strongest_series']
                st.metric("Strongest Series", f"{strongest}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                weakest = series_analysis['weakest_series']
                st.metric("Weakest Series", f"{weakest}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Series Performance Breakdown
            st.markdown("#### 🎯 Series-by-Series Performance Analysis")
            
            # Create columns for detailed series analysis
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("##### 📈 Improving Series")
                if series_analysis['improving_series']:
                    for series_num in series_analysis['improving_series']:
                        prog = series_analysis['series_progression'][series_num-1]
                        st.markdown(f"**Series {series_num}**: {prog['starting_score']:.1f} → {prog['current_score']:.1f} "
                                  f"<span class='trend-positive'>(+{prog['improvement']:.2f})</span>", 
                                  unsafe_allow_html=True)
                else:
                    st.info("No series showing significant improvement")
                
                st.markdown("##### ➡️ Stable Series")
                if series_analysis['stable_series']:
                    for series_num in series_analysis['stable_series']:
                        prog = series_analysis['series_progression'][series_num-1]
                        st.markdown(f"**Series {series_num}**: {prog['starting_score']:.1f} → {prog['current_score']:.1f} "
                                  f"<span class='trend-neutral'>({prog['improvement']:+.2f})</span>", 
                                  unsafe_allow_html=True)
                else:
                    st.info("No stable series")
            
            with analysis_col2:
                st.markdown("##### 📉 Declining Series")
                if series_analysis['declining_series']:
                    for series_num in series_analysis['declining_series']:
                        prog = series_analysis['series_progression'][series_num-1]
                        st.markdown(f"**Series {series_num}**: {prog['starting_score']:.1f} → {prog['current_score']:.1f} "
                                  f"<span class='trend-negative'>({prog['improvement']:.2f})</span>", 
                                  unsafe_allow_html=True)
                else:
                    st.success("No series showing significant decline!")
                
                st.markdown("##### 🎯 Consistency Analysis")
                st.markdown(f"**Most Consistent**: Series {series_analysis['most_consistent_series']}")
                st.markdown(f"**Least Consistent**: Series {series_analysis['least_consistent_series']}")
            
            # Create and display trend visualizations
            st.markdown("#### 📊 Series Performance Visualizations")
            
            tab1, tab2 = st.tabs(["Basic Trends", "Detailed Progression"])
            
            with tab1:
                st.markdown("##### Basic Series Trends")
                fig1 = predictor.create_series_trend_plot(sessions_data, series_analysis)
                st.pyplot(fig1)
            
            with tab2:
                st.markdown("##### Detailed Series Progression")
                fig2 = predictor.create_series_progression_plot(sessions_data, series_analysis)
                st.pyplot(fig2)
                
                # Additional insights
                st.markdown("##### 🔍 Key Insights from Progression Analysis")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("**Performance Patterns:**")
                    if series_analysis['endurance_factor'] > 0:
                        st.success("• Strong finisher - performs better in later series")
                    else:
                        st.warning("• Needs endurance work - performance drops in later series")
                    
                    if len(series_analysis['improving_series']) >= 3:
                        st.success("• Multiple series showing positive development")
                    elif len(series_analysis['declining_series']) >= 3:
                        st.error("• Multiple series showing concerning decline")
                    
                    avg_improvement = np.mean([p['improvement'] for p in series_analysis['series_progression']])
                    if avg_improvement > 0.3:
                        st.success(f"• Overall positive trend across all series (+{avg_improvement:.2f})")
                    elif avg_improvement < -0.3:
                        st.error(f"• Overall declining trend across all series ({avg_improvement:.2f})")
                
                with insights_col2:
                    st.markdown("**Training Focus Areas:**")
                    if series_analysis['weakest_series']:
                        st.info(f"• Priority focus on Series {series_analysis['weakest_series']} (lowest average)")
                    if series_analysis['least_consistent_series']:
                        st.info(f"• Stabilize Series {series_analysis['least_consistent_series']} (most variable)")
                    if series_analysis['declining_series']:
                        declining_list = ", ".join([str(s) for s in series_analysis['declining_series']])
                        st.warning(f"• Address decline in Series {declining_list}")
                    if series_analysis['improving_series']:
                        improving_list = ", ".join([str(s) for s in series_analysis['improving_series']])
                        st.success(f"• Maintain positive momentum in Series {improving_list}")
    
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
            
            if series_analysis:
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
                    # Use series analysis for recommendations
                    weakest = series_analysis['weakest_series']
                    strongest = series_analysis['strongest_series']
                    least_consistent = series_analysis['least_consistent_series']
                    endurance = series_analysis['endurance_factor']
                    
                    st.write(f"• 💪 Strengthen Series {weakest} (lowest average)")
                    st.write(f"• ✅ Maintain Series {strongest} (highest average)")
                    st.write(f"• 🎯 Stabilize Series {least_consistent} (most variable)")
                    
                    if endurance < -0.5:
                        st.write("• ⚡ Build physical and mental endurance")
                        st.write("• 🏋️ Focus on maintaining focus in later series")
                    elif endurance > 0.5:
                        st.write("• 🔥 Leverage strong finishing ability")
                    
                    st.write("• 📈 Practice series transitions smoothly")
        
        else:
            st.error("Please complete all 3 sessions with 6 series each")

    # Example section
    with st.expander("💡 How to use this predictor"):
        st.write("""
        **Enhanced Series Analysis Features:**
        
        **Series Progress Tracking:**
        - See which series are improving, declining, or stable
        - Track progression from Session 1 to Session 3 for each series
        - Identify consistency patterns across different series
        
        **Visual Analysis:**
        - Series progression graphs showing performance over sessions
        - Improvement/decline analysis for each series
        - Current performance vs average comparison
        - Heatmap visualization of all series across sessions
        
        **Key Metrics:**
        - Endurance factor (late vs early series performance)
        - Series consistency analysis
        - Improvement trends per series
        - Strongest and weakest series identification
        
        **Example Insights:**
        - "Series 3 improved by +1.2 points from Session 1 to Session 3"
        - "Series 5 is most consistent with low variability"
        - "Performance drops in later series - focus on endurance"
        - "Series 2 shows concerning decline - needs attention"
        """)

if __name__ == "__main__":
    main()