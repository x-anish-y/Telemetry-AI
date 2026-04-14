import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { 
  Wind, Activity, AlertTriangle, CheckCircle, 
  Settings, LogOut, Plus, Zap, MapPin, Clock,
  TrendingUp, Gauge
} from 'lucide-react';
import { toast } from 'sonner';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const getStatusInfo = (detection) => {
  if (!detection) return { status: 'offline', color: 'slate', icon: Clock };
  
  if (detection.fault_type === 'normal' || !detection.detected) {
    return { status: 'healthy', color: 'emerald', icon: CheckCircle };
  }
  
  if (detection.severity > 0.7) {
    return { status: 'critical', color: 'rose', icon: AlertTriangle };
  }
  
  return { status: 'warning', color: 'amber', icon: Activity };
};

export default function Dashboard() {
  const [turbines, setTurbines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [seeding, setSeeding] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    fetchTurbines();
  }, []);

  const fetchTurbines = async () => {
    try {
      const res = await axios.get(`${API}/turbines`);
      setTurbines(res.data);
    } catch (err) {
      console.error('Failed to fetch turbines:', err);
      toast.error('Failed to load turbines');
    } finally {
      setLoading(false);
    }
  };

  const seedDemoData = async () => {
    setSeeding(true);
    try {
      await axios.post(`${API}/seed`);
      toast.success('Demo turbines created!');
      fetchTurbines();
    } catch (err) {
      toast.error('Failed to seed data');
    } finally {
      setSeeding(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const stats = {
    total: turbines.length,
    healthy: turbines.filter(t => getStatusInfo(t.last_detection).status === 'healthy').length,
    warning: turbines.filter(t => getStatusInfo(t.last_detection).status === 'warning').length,
    critical: turbines.filter(t => getStatusInfo(t.last_detection).status === 'critical').length,
  };

  return (
    <div className="min-h-screen bg-[#020617] grid-bg">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-sm flex items-center justify-center border border-cyan-500/30">
                <Wind className="w-5 h-5 text-cyan-500" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-100 tracking-tight">Telemetry AI</h1>
                <p className="text-xs text-slate-500">Predictive Maintenance Dashboard</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-sm text-slate-400 hidden sm:block">
                {user?.name || user?.email}
              </span>
              <Link to="/simulator">
                <Button variant="outline" size="sm" data-testid="simulator-link" className="border-slate-700 text-slate-300 hover:bg-slate-800">
                  <Activity className="w-4 h-4 mr-2" />
                  Simulator
                </Button>
              </Link>
              <Link to="/admin">
                <Button variant="outline" size="sm" data-testid="admin-link" className="border-slate-700 text-slate-300 hover:bg-slate-800">
                  <Settings className="w-4 h-4" />
                </Button>
              </Link>
              <Button variant="ghost" size="sm" onClick={handleLogout} data-testid="logout-btn" className="text-slate-400 hover:text-slate-200">
                <LogOut className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-slate-900 border-slate-800" data-testid="stat-total">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Total Turbines</p>
                  <p className="metric-value text-3xl text-slate-100">{stats.total}</p>
                </div>
                <Gauge className="w-8 h-8 text-cyan-500 opacity-50" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900 border-slate-800" data-testid="stat-healthy">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Healthy</p>
                  <p className="metric-value text-3xl text-emerald-500">{stats.healthy}</p>
                </div>
                <CheckCircle className="w-8 h-8 text-emerald-500 opacity-50" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900 border-slate-800" data-testid="stat-warning">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Warning</p>
                  <p className="metric-value text-3xl text-amber-500">{stats.warning}</p>
                </div>
                <Activity className="w-8 h-8 text-amber-500 opacity-50" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900 border-slate-800" data-testid="stat-critical">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Critical</p>
                  <p className="metric-value text-3xl text-rose-500">{stats.critical}</p>
                </div>
                <AlertTriangle className="w-8 h-8 text-rose-500 opacity-50" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-200">Wind Turbines</h2>
          <div className="flex gap-2">
            {turbines.length === 0 && (
              <Button 
                onClick={seedDemoData} 
                disabled={seeding}
                data-testid="seed-demo-btn"
                className="bg-cyan-600 hover:bg-cyan-500 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                {seeding ? 'Seeding...' : 'Add Demo Turbines'}
              </Button>
            )}
          </div>
        </div>

        {/* Turbine Grid */}
        {loading ? (
          <div className="text-center py-12 text-slate-500">Loading turbines...</div>
        ) : turbines.length === 0 ? (
          <Card className="bg-slate-900 border-slate-800 border-dashed">
            <CardContent className="py-12 text-center">
              <Wind className="w-12 h-12 mx-auto text-slate-600 mb-4" />
              <p className="text-slate-400">No turbines configured yet</p>
              <p className="text-slate-500 text-sm mt-1">Click "Add Demo Turbines" to get started</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {turbines.map((turbine) => {
              const { status, color, icon: StatusIcon } = getStatusInfo(turbine.last_detection);
              
              return (
                <Card 
                  key={turbine.id}
                  className={`bg-slate-900 border-slate-800 card-interactive cursor-pointer`}
                  onClick={() => navigate(`/turbines/${turbine.id}`)}
                  data-testid={`turbine-card-${turbine.id}`}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-sm flex items-center justify-center bg-${color}-500/10 border border-${color}-500/30`}>
                          <Wind className={`w-5 h-5 text-${color}-500`} />
                        </div>
                        <div>
                          <CardTitle className="text-lg text-slate-100">{turbine.name}</CardTitle>
                          <div className="flex items-center gap-1 text-xs text-slate-500 mt-1">
                            <MapPin className="w-3 h-3" />
                            {turbine.location}
                          </div>
                        </div>
                      </div>
                      <Badge 
                        variant="outline" 
                        className={`status-${status === 'healthy' ? 'ok' : status === 'warning' ? 'warning' : status === 'critical' ? 'critical' : 'offline'} border`}
                      >
                        <StatusIcon className="w-3 h-3 mr-1" />
                        {status.toUpperCase()}
                      </Badge>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-500 flex items-center gap-1">
                        <Zap className="w-3 h-3" />
                        Rated Power
                      </span>
                      <span className="font-mono text-slate-300">{turbine.rated_power_kw} kW</span>
                    </div>

                    {turbine.last_detection && (
                      <>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-slate-500">Fault Type</span>
                          <span className="font-mono text-slate-300 capitalize">
                            {turbine.last_detection.fault_type?.replace('_', ' ') || 'N/A'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-slate-500">Severity</span>
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-2 bg-slate-800 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${turbine.last_detection.severity > 0.7 ? 'bg-rose-500' : turbine.last_detection.severity > 0.4 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                style={{ width: `${(turbine.last_detection.severity || 0) * 100}%` }}
                              />
                            </div>
                            <span className="font-mono text-slate-300 w-10 text-right">
                              {((turbine.last_detection.severity || 0) * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </>
                    )}

                    <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
                      <span className="text-xs text-slate-600">
                        {turbine.last_detection 
                          ? `Last scan: ${new Date(turbine.last_detection.created_at).toLocaleDateString()}`
                          : 'No scans yet'
                        }
                      </span>
                      <TrendingUp className="w-4 h-4 text-slate-600" />
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
