"""
Interactive Privacy Audit Dashboard using Plotly Dash.
"""
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
from pathlib import Path

class PrivacyDashboard:
    """
    Interactive dashboard for privacy audit visualization.
    """
    
    def __init__(self, audit_dir: str = "outputs/phase3"):
        self.audit_dir = Path(audit_dir)
        self.app = Dash(__name__)
        self.load_data()
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self):
        """Load audit results"""
        # Load main audit report
        with open(self.audit_dir / "audit_report.json") as f:
            self.audit_report = json.load(f)
        
        # Load results CSV
        self.results_df = pd.read_csv(self.audit_dir / "audit_results.csv")
        
        # Load compliance report
        with open(self.audit_dir / "compliance_report.json") as f:
            self.compliance = json.load(f)
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("LLM Privacy Audit Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Summary cards
            html.Div([
                self._create_metric_card("Total Samples", 
                                        self.audit_report['metadata']['num_samples']),
                self._create_metric_card("High-Risk Samples", 
                                        len(self.audit_report['high_risk_samples'])),
                self._create_metric_card("Compliance Score", 
                                        f"{self.compliance['compliance_summary']['overall_score']:.1f}%"),
                self._create_metric_card("Filter Rate", 
                                        f"{self.audit_report['privacy_loss']['filter_rate']:.1%}")
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
            
            # Risk distribution
            html.Div([
                html.H2("Risk Score Distribution"),
                dcc.Graph(id='risk-distribution')
            ], style={'margin': '20px'}),
            
            # Attack comparison
            html.Div([
                html.H2("Attack Method Comparison"),
                dcc.Graph(id='attack-comparison')
            ], style={'margin': '20px'}),
            
            # Compliance breakdown
            html.Div([
                html.H2("Regulatory Compliance"),
                dcc.Graph(id='compliance-breakdown')
            ], style={'margin': '20px'}),
            
            # High-risk samples table
            html.Div([
                html.H2("Top 10 High-Risk Samples"),
                html.Div(id='high-risk-table')
            ], style={'margin': '20px'}),
            
            # Checkpoint history (if available)
            html.Div([
                html.H2("Memorization Over Training"),
                dcc.Graph(id='checkpoint-history')
            ], style={'margin': '20px'})
        ])
    
    def _create_metric_card(self, title, value):
        """Create a metric card"""
        return html.Div([
            html.H4(title, style={'color': '#7f8c8d'}),
            html.H2(str(value), style={'color': '#2c3e50', 'margin': '10px'})
        ], style={
            'border': '2px solid #ecf0f1',
            'borderRadius': '10px',
            'padding': '20px',
            'textAlign': 'center',
            'minWidth': '200px'
        })
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            Output('risk-distribution', 'figure'),
            Input('risk-distribution', 'id')
        )
        def update_risk_distribution(_):
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=self.results_df['ensemble_risk'],
                nbinsx=30,
                name='Ensemble Risk',
                marker_color='#3498db'
            ))
            
            fig.update_layout(
                xaxis_title="Risk Score",
                yaxis_title="Count",
                showlegend=True,
                template="plotly_white"
            )
            
            # Add threshold line
            threshold = 0.7
            fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                         annotation_text="Threshold")
            
            return fig
        
        @self.app.callback(
            Output('attack-comparison', 'figure'),
            Input('attack-comparison', 'id')
        )
        def update_attack_comparison(_):
            # Compare different attack scores
            attack_cols = ['risk_score', 'token_loss_risk', 'mink_prob_risk', 'neighborhood_risk']
            attack_names = ['Perplexity', 'Token Loss', 'Min-K Prob', 'Neighborhood']
            
            fig = go.Figure()
            
            for col, name in zip(attack_cols, attack_names):
                if col in self.results_df.columns:
                    fig.add_trace(go.Box(
                        y=self.results_df[col],
                        name=name,
                        boxmean='sd'
                    ))
            
            fig.update_layout(
                yaxis_title="Risk Score",
                showlegend=True,
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output('compliance-breakdown', 'figure'),
            Input('compliance-breakdown', 'id')
        )
        def update_compliance(_):
            requirements = self.compliance['requirements']
            
            # Count by regulation
            reg_counts = {}
            for req in requirements:
                reg = req['regulation']
                status = req['status']
                if reg not in reg_counts:
                    reg_counts[reg] = {'met': 0, 'partial': 0, 'not_met': 0}
                reg_counts[reg][status] += 1
            
            # Create stacked bar chart
            regulations = list(reg_counts.keys())
            met = [reg_counts[r]['met'] for r in regulations]
            partial = [reg_counts[r]['partial'] for r in regulations]
            not_met = [reg_counts[r]['not_met'] for r in regulations]
            
            fig = go.Figure(data=[
                go.Bar(name='Met', x=regulations, y=met, marker_color='#2ecc71'),
                go.Bar(name='Partial', x=regulations, y=partial, marker_color='#f39c12'),
                go.Bar(name='Not Met', x=regulations, y=not_met, marker_color='#e74c3c')
            ])
            
            fig.update_layout(
                barmode='stack',
                yaxis_title="Requirements",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output('high-risk-table', 'children'),
            Input('high-risk-table', 'id')
        )
        def update_high_risk_table(_):
            high_risk = sorted(
                self.audit_report['high_risk_samples'],
                key=lambda x: x['ensemble_risk'],
                reverse=True
            )[:10]
            
            table_rows = []
            for sample in high_risk:
                table_rows.append(html.Tr([
                    html.Td(sample['sample_id']),
                    html.Td(sample['text'][:100] + "..."),
                    html.Td(f"{sample['ensemble_risk']:.3f}"),
                    html.Td(f"{sample['perplexity']:.2f}")
                ]))
            
            return html.Table([
                html.Thead(html.Tr([
                    html.Th("ID"),
                    html.Th("Text"),
                    html.Th("Risk Score"),
                    html.Th("Perplexity")
                ])),
                html.Tbody(table_rows)
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'border': '1px solid #ddd'
            })
        
        @self.app.callback(
            Output('checkpoint-history', 'figure'),
            Input('checkpoint-history', 'id')
        )
        def update_checkpoint_history(_):
            # Try to load checkpoint audit history
            checkpoint_file = Path("outputs/checkpoint_audits/audit_history.csv")
            
            if checkpoint_file.exists():
                df = pd.read_csv(checkpoint_file)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['step'],
                    y=df['high_risk_rate'],
                    mode='lines+markers',
                    name='High-Risk Rate',
                    line=dict(color='#e74c3c', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['step'],
                    y=df['mean_risk_score'],
                    mode='lines+markers',
                    name='Mean Risk Score',
                    line=dict(color='#3498db', width=2),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    xaxis_title="Training Step",
                    yaxis_title="High-Risk Rate",
                    yaxis2=dict(
                        title="Mean Risk Score",
                        overlaying='y',
                        side='right'
                    ),
                    template="plotly_white"
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="No checkpoint audit history available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            return fig
    
    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        print(f"\n{'='*60}")
        print("  PRIVACY AUDIT DASHBOARD")
        print(f"{'='*60}\n")
        print(f"Starting dashboard at http://localhost:{port}")
        print("Press Ctrl+C to stop\n")
        
        self.app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_dir", default="outputs/phase3")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    dashboard = PrivacyDashboard(args.audit_dir)
    dashboard.run(debug=args.debug, port=args.port)