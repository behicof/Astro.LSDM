import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import threading
import MetaTrader5 as mt5

class AstroTraderDashboard:
    def __init__(self, model, astro_collector, trade_executor):
        self.model = model
        self.astro_collector = astro_collector
        self.trade_executor = trade_executor
        self.stop_thread = False
        self.signal_thread = None
        self.latest_signal = None
        
    def start_signal_thread(self):
        """Start a thread that periodically generates signals"""
        self.stop_thread = False
        
        def signal_loop():
            while not self.stop_thread:
                try:
                    self.latest_signal = self.trade_executor.generate_signal()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    print(f"Error in signal thread: {str(e)}")
                    time.sleep(60)
        
        self.signal_thread = threading.Thread(target=signal_loop)
        self.signal_thread.daemon = True
        self.signal_thread.start()
        
    def stop_signal_thread(self):
        """Stop the signal generation thread"""
        self.stop_thread = True
        if self.signal_thread:
            self.signal_thread.join(timeout=2)
    
    def get_planet_positions_chart(self):
        """Generate a chart showing current planet positions in the zodiac"""
        # Get current time
        ts = load.timescale()
        now = datetime.now()
        t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)
        
        planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Rahu', 'Ketu']
        
        # Get positions
        positions = []
        for planet in planets:
            if planet == 'Rahu' or planet == 'Ketu':
                lon = self.astro_collector.get_node_positions(t)[0 if planet == 'Rahu' else 1]
            else:
                lon = self.astro_collector.get_planet_longitude(planet.lower(), t)
            positions.append(lon)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Define zodiac signs and their ranges
        zodiac_signs = [
            'Aries', 'Taurus', 'Gemini', 'Cancer', 
            'Leo', 'Virgo', 'Libra', 'Scorpio', 
            'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
        ]
        
        # Create zodiac wheel
        for i, sign in enumerate(zodiac_signs):
            start_angle = i * 30
            end_angle = (i + 1) * 30
            
            # Create slice for each zodiac sign
            fig.add_shape(
                type="path",
                path=f"M 0.5,0.5 L {0.5 + 0.4*np.cos(np.radians(start_angle))},{0.5 + 0.4*np.sin(np.radians(start_angle))} A 0.4,0.4 0 0,1 {0.5 + 0.4*np.cos(np.radians(end_angle))},{0.5 + 0.4*np.sin(np.radians(end_angle))} Z",
                fillcolor=f"rgba({(i*20)%255},{(i*40)%255},{(i*60)%255},0.2)",
                line=dict(color="white", width=1),
            )
            
            # Add text for zodiac sign
            middle_angle = (start_angle + end_angle) / 2
            fig.add_annotation(
                x=0.5 + 0.45*np.cos(np.radians(middle_angle)),
                y=0.5 + 0.45*np.sin(np.radians(middle_angle)),
                text=sign,
                showarrow=False,
                font=dict(size=10)
            )
        
        # Plot planets on wheel
        for i, (planet, pos) in enumerate(zip(planets, positions)):
            angle = pos
            r = 0.3  # Radius for plotting planets
            
            x = 0.5 + r*np.cos(np.radians(angle))
            y = 0.5 + r*np.sin(np.radians(angle))
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=12, color=f"hsl({(i*30)%360},80%,50%)"),
                text=[planet],
                textposition="top center",
                name=planet
            ))
        
        # Configure layout
        fig.update_layout(
            title="Current Planet Positions in Zodiac",
            showlegend=False,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1]),
            width=700, height=700,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def get_aspects_table(self):
        """Generate a table of current planetary aspects"""
        # Get current time
        ts = load.timescale()
        now = datetime.now()
        t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)
        
        # Get aspects
        aspects = self.astro_collector.calculate_aspects(t)
        
        # Convert to readable format
        aspect_list = []
        for key, value in aspects.items():
            if value == 1:
                planets, aspect_type, planet2 = key.split('_')
                aspect_list.append({
                    "Planet 1": planets,
                    "Aspect": aspect_type.capitalize(),
                    "Planet 2": planet2,
                    "Strength": 1.0  # Could calculate aspect strength based on orb
                })
                
        return pd.DataFrame(aspect_list)
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        # Set page config
        st.set_page_config(page_title="OmniPattern AstroTrader Pro", layout="wide")
        
        # Title
        st.title("OmniPattern AstroTrader Pro - XAUUSD")
        
        # Sidebar
        st.sidebar.header("Control Panel")
        
        # Trading controls
        st.sidebar.subheader("Trading Controls")
        
        if st.sidebar.button("Start Auto Trading"):
            self.start_signal_thread()
            st.sidebar.success("Auto trading started!")
            
        if st.sidebar.button("Stop Auto Trading"):
            self.stop_signal_thread()
            st.sidebar.error("Auto trading stopped!")
            
        if st.sidebar.button("Close All Positions"):
            self.trade_executor.close_all_positions()
            st.sidebar.warning("All positions closed!")
        
        # Main dashboard layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Current Signal
            st.subheader("Current Trading Signal")
            
            if self.latest_signal:
                signal = self.latest_signal['signal']
                confidence = self.latest_signal['confidence']
                reversal_prob = self.latest_signal['reversal_probability']
                
                # Signal indicator
                if signal == "BUY":
                    st.markdown(f"### 🟢 BUY ({confidence:.2f})")
                elif signal == "SELL":
                    st.markdown(f"### 🔴 SELL ({confidence:.2f})")
                else:
                    st.markdown(f"### ⚪ HOLD ({confidence:.2f})")
                
                # Current price and prediction
                st.metric(
                    label="XAUUSD Price", 
                    value=f"${self.latest_signal['current_price']:.2f}",
                    delta=f"{self.latest_signal['predicted_move']:.2f}"
                )
                
                # Reversal probability gauge
                st.markdown(f"### Trend Reversal Probability: {reversal_prob:.2%}")
                
                # Plot gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=reversal_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Reversal Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
                
            else:
                st.info("Signal not available. Start auto trading to generate signals.")
            
            # Current planetary aspects
            st.subheader("Current Planetary Aspects")
            aspects_df = self.get_aspects_table()
            
            if not aspects_df.empty:
                st.dataframe(aspects_df, hide_index=True)
            else:
                st.info("No significant aspects at this time.")
                
        with col2:
            # Planet positions
            st.subheader("Current Planet Positions")
            planet_chart = self.get_planet_positions_chart()
            st.plotly_chart(planet_chart)
            
            # Price chart with predictions
            st.subheader("XAUUSD Price Chart with Predictions")
            
            # Get recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # 7 days of data
            price_data = self.astro_collector.get_xauusd_data(start_time, end_time)
            
            # Create plotly candlestick chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               row_heights=[0.7, 0.3],
                               specs=[[{"type": "candlestick"}],
                                      [{"type": "bar"}]])
            
            fig.add_trace(
                go.Candlestick(
                    x=price_data['time'],
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'],
                    name="XAUUSD"
                ),
                row=1, col=1
            )
            
            # Add volume
            fig.add_trace(
                go.Bar(
                    x=price_data['time'],
                    y=price_data['tick_volume'],
                    name="Volume",
                    marker_color='rgba(0,0,255,0.5)'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title="XAUUSD Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Bottom section for overall market analysis
        st.subheader("Key Astrological Insights")
        
        # Create two columns
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            ### Upcoming Astrological Events
            
            | Date | Event | Expected Impact |
            | --- | --- | --- |
            | 2025-04-22 | Venus Trine Jupiter | Strong buying opportunity |
            | 2025-05-10 | Saturn Square Moon | Potential reversal pattern |
            | 2025-05-18 | Mars enters Taurus | High volatility expected |
            """)
        
        with col4:
            st.markdown("""
            ### Active Planetary Influences on Gold
            
            * 🔵 **Jupiter in Taurus** - Traditionally bullish for gold prices
            * 🔴 **Saturn in Aquarius** - Possible resistance at key technical levels
            * 🟠 **Rahu near Uranus** - Increased possibility of market surprises
            * 🟢 **Venus-Mars alignment** - Potential for sudden upside movements
            """)
        
        # Show raw astrological data (collapsible)
        with st.expander("Show Raw Astrological Data"):
            # Get current time
            ts = load.timescale()
            now = datetime.now()
            t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)
            
            # Get planetary positions
            planets = ['sun', 'moon', 'mercury', 'venus', 'mars', 
                      'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
            
            positions = {}
            for planet in planets:
                lon = self.astro_collector.get_planet_longitude(planet, t)
                sign = zodiac_signs[int(lon / 30)]
                deg = lon % 30
                
                # Check if retrograde
                retrograde = self.astro_collector.is_retrograde(planet, t)
                
                positions[planet.capitalize()] = {
                    "Longitude": f"{lon:.2f}°",
                    "Sign": sign,
                    "Position": f"{int(deg)}° {sign}",
                    "Retrograde": "Yes" if retrograde else "No"
                }
            
            # Display as dataframe
            positions_df = pd.DataFrame(positions).T
            st.dataframe(positions_df)
        
        # Refresh data automatically
        st.empty()
        time.sleep(10)
        st.experimental_rerun()