import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Switch } from '../components/ui/switch';
import { Label } from '../components/ui/label';
import { ArrowLeft, Settings, Shield, Server, AlertCircle, CheckCircle, Brain, BarChart3 } from 'lucide-react';
import { toast } from 'sonner';

const API = process.env.REACT_APP_BACKEND_URL
  ? `${process.env.REACT_APP_BACKEND_URL}/api`
  : '/api';

export default function Admin() {
  const [config, setConfig] = useState({ demo_mode: true, ml_service_url: 'internal', ml_status: 'unknown' });
  const [mlStatus, setMlStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchConfig();
    fetchMlStatus();
  }, []);

  const fetchConfig = async () => {
    try {
      const res = await axios.get(`${API}/config`);
      setConfig(res.data);
    } catch (err) {
      toast.error('Failed to load config');
    } finally {
      setLoading(false);
    }
  };

  const fetchMlStatus = async () => {
    try {
      const res = await axios.get(`${API}/ml-status`);
      setMlStatus(res.data);
    } catch (err) {
      console.error('Failed to fetch ML status:', err);
    }
  };

  const updateDemoMode = async (enabled) => {
    setSaving(true);
    try {
      const res = await axios.put(`${API}/config`, { demo_mode: enabled });
      setConfig({ ...config, demo_mode: res.data.demo_mode });
      toast.success(`Demo mode ${enabled ? 'enabled' : 'disabled'}`);
    } catch (err) {
      toast.error('Failed to update config');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#020617] flex items-center justify-center">
        <div className="text-cyan-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#020617] grid-bg">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center gap-4">
            <Link to="/dashboard">
              <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200" data-testid="back-btn">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
            </Link>
            <div>
              <h1 className="text-xl font-bold text-slate-100">Admin Settings</h1>
              <p className="text-xs text-slate-500">System configuration and demo mode</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Demo Mode */}
        <Card className="bg-slate-900 border-slate-800" data-testid="demo-mode-card">
          <CardHeader>
            <CardTitle className="text-slate-200 flex items-center gap-2">
              <Shield className="w-5 h-5 text-cyan-500" />
              Demo Mode
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-slate-950 rounded-sm border border-slate-800">
              <div>
                <Label className="text-slate-300 font-medium">Enable Demo Fallback</Label>
                <p className="text-xs text-slate-500 mt-1">
                  When enabled, uses deterministic mock data if ML service is unavailable
                </p>
              </div>
              <Switch
                checked={config.demo_mode}
                onCheckedChange={updateDemoMode}
                disabled={saving}
                data-testid="demo-mode-switch"
              />
            </div>

            <div className={`flex items-center gap-3 p-3 rounded-sm border ${config.demo_mode ? 'bg-amber-500/10 border-amber-500/30' : 'bg-emerald-500/10 border-emerald-500/30'}`}>
              {config.demo_mode ? (
                <>
                  <AlertCircle className="w-5 h-5 text-amber-500" />
                  <div>
                    <p className="text-amber-400 text-sm font-medium">Demo Mode Active</p>
                    <p className="text-amber-500/70 text-xs">ML analysis will use fallback data if service unreachable</p>
                  </div>
                </>
              ) : (
                <>
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                  <div>
                    <p className="text-emerald-400 text-sm font-medium">Production Mode</p>
                    <p className="text-emerald-500/70 text-xs">Real ML analysis only, no fallback</p>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ML Service */}
        <Card className="bg-slate-900 border-slate-800" data-testid="ml-service-card">
          <CardHeader>
            <CardTitle className="text-slate-200 flex items-center gap-2">
              <Brain className="w-5 h-5 text-cyan-500" />
              ML Engine Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="p-4 bg-slate-950 rounded-sm border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-slate-500 text-sm">Status</span>
                <span className={`flex items-center gap-2 text-sm ${mlStatus?.status === 'ready' ? 'text-emerald-400' : 'text-amber-400'}`}>
                  <span className={`w-2 h-2 rounded-full ${mlStatus?.status === 'ready' ? 'bg-emerald-500' : 'bg-amber-500'} animate-pulse`} />
                  {mlStatus?.status === 'ready' ? 'ML Models Loaded' : 'Checking...'}
                </span>
              </div>
              {mlStatus?.models_loaded && (
                <div className="pt-2 border-t border-slate-800">
                  <p className="text-xs text-slate-500 mb-2">Loaded Models:</p>
                  <div className="space-y-1">
                    {mlStatus.models_loaded.map((model, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs">
                        <CheckCircle className="w-3 h-3 text-emerald-500" />
                        <span className="text-slate-400">{model}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <p className="text-xs text-slate-600 pt-2 border-t border-slate-800">
                Real ML analysis using trained Isolation Forest, Random Forest, and Gradient Boosting models.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Feature Importance */}
        {mlStatus?.feature_importance && (
          <Card className="bg-slate-900 border-slate-800" data-testid="feature-importance-card">
            <CardHeader>
              <CardTitle className="text-slate-200 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-cyan-500" />
                Top ML Features
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {mlStatus.feature_importance.features.slice(0, 8).map((feature, idx) => (
                  <div key={feature} className="flex items-center gap-3">
                    <span className="text-xs text-slate-500 w-4">{idx + 1}.</span>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-slate-400 font-mono">{feature}</span>
                        <span className="text-xs text-cyan-400 font-mono">
                          {(mlStatus.feature_importance.importance[idx] * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-cyan-500 rounded-full"
                          style={{ width: `${mlStatus.feature_importance.importance[idx] * 100 * 8}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* System Info */}
        <Card className="bg-slate-900 border-slate-800" data-testid="system-info-card">
          <CardHeader>
            <CardTitle className="text-slate-200 flex items-center gap-2">
              <Settings className="w-5 h-5 text-cyan-500" />
              System Info
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-slate-950 rounded-sm border border-slate-800">
                <p className="metric-label">Backend</p>
                <p className="text-slate-300 font-medium">FastAPI + MongoDB</p>
              </div>
              <div className="p-3 bg-slate-950 rounded-sm border border-slate-800">
                <p className="metric-label">Frontend</p>
                <p className="text-slate-300 font-medium">React + Tailwind</p>
              </div>
              <div className="p-3 bg-slate-950 rounded-sm border border-slate-800">
                <p className="metric-label">ML Engine</p>
                <p className="text-slate-300 font-medium">scikit-learn</p>
              </div>
              <div className="p-3 bg-slate-950 rounded-sm border border-slate-800">
                <p className="metric-label">PDF Engine</p>
                <p className="text-slate-300 font-medium">ReportLab</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
