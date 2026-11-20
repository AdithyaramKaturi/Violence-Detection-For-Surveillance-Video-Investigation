import codecs

with codecs.open('app.py', 'r', 'utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if 'fig = go.Figure()' in line and '# Build binary step plot' in lines[i-1]:
        start_idx = i - 1
    if 'st.plotly_chart(fig, use_container_width=True)' in line and start_idx != -1:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_code = '''                # Build continuous line chart with colored segments
                fig = go.Figure()
                
                current_state = None
                seg_x = []
                seg_y = []
                
                for _, row in timeline_df.iterrows():
                    is_violent = row["is_violent"]
                    x_val = row["time_sec"]
                    y_val = row["confidence"]
                    
                    if current_state is None:
                        current_state = is_violent
                        
                    if is_violent != current_state:
                        color = 'red' if current_state else '#1f77b4'
                        fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines', line=dict(color=color, width=2), showlegend=False))
                        
                        current_state = is_violent
                        seg_x = [seg_x[-1], x_val] if seg_x else [x_val]
                        seg_y = [seg_y[-1], y_val] if seg_y else [y_val]
                    else:
                        seg_x.append(x_val)
                        seg_y.append(y_val)
                        
                if seg_x:
                    color = 'red' if current_state else '#1f77b4'
                    fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines', line=dict(color=color, width=2), showlegend=False))

                # Add a subtle threshold line
                fig.add_hline(y=threshold_pct, line_dash="dash", line_color="#FF5050", opacity=0.5)

                fig.update_layout(
                    xaxis=dict(title="Video Time (seconds)", tickmode='array', tickvals=tick_vals, ticktext=tick_text),
                    yaxis=dict(title="Confidence (%)", range=[-5, 105]),
                    height=250,
                    margin=dict(l=40, r=20, t=20, b=40)
                )
'''
    lines[start_idx:end_idx] = [new_code]
    with codecs.open('app.py', 'w', 'utf-8') as f:
        f.writelines(lines)
    print("Replaced successfully.")
else:
    print("Could not find block.")
