import { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import {
  Wind, ArrowLeft, Upload, FileText, Download,
  AlertTriangle, CheckCircle, Activity, Zap, MapPin,
  TrendingUp, TrendingDown, DollarSign, Wrench, Clock,
  ThermometerSun, Gauge, RotateCw, Info, FileDown
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, Legend
} from 'recharts';
import { toast } from 'sonner';
import html2canvas from 'html2canvas';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Expected CSV format information
const CSV_FORMAT = {
  required: [
    { name: 'timestamp', desc: 'ISO datetime', example: '2024-01-01 00:00:00' },
    { name: 'wind_speed', desc: 'Wind velocity (m/s)', example: '12.5' },
    { name: 'rotor_rpm', desc: 'Rotor speed (RPM)', example: '14.2' },
    { name: 'power', desc: 'Active power (kW)', example: '2450' },
    { name: 'pitch_cmd', desc: 'Commanded pitch (°)', example: '5.2' },
    { name: 'pitch_meas', desc: 'Measured pitch (°)', example: '5.4' },
    { name: 'torque', desc: 'Generator torque (kNm)', example: '125.3' },
    { name: 'vibration', desc: 'Nacelle vibration (m/s²)', example: '0.65' },
  ],
  optional: [
    { name: 'temperature', desc: 'Gearbox/Nacelle temp (°C)', example: '45.2' },
    { name: 'rotor_phase', desc: 'Rotor azimuth (0-360°)', example: '180.5' },
  ]
};

// Remediation recommendations based on fault type
const RECOMMENDATIONS = {
  erosion: {
    title: 'Blade Surface Erosion Detected',
    urgency: 'MEDIUM',
    actions: [
      'Schedule visual inspection via drone imagery within 7 days',
      'Assess leading edge protection (LEP) tape condition',
      'Plan blade repair during next maintenance window',
      'Apply protective coating after cleaning',
    ],
    parts: [
      { name: 'Blade Repair Kit (LEP)', quantity: 1, cost: 35000 },
      { name: 'Protective Coating', quantity: 2, cost: 8000 },
      { name: 'Surface Cleaner', quantity: 1, cost: 2500 },
    ],
    laborHours: 16,
  },
  pitch_drift: {
    title: 'Pitch System Malfunction Detected',
    urgency: 'HIGH',
    actions: [
      'Immediate pitch system calibration required',
      'Check pitch motor encoder alignment',
      'Inspect pitch bearing for wear',
      'Verify pitch controller firmware version',
    ],
    parts: [
      { name: 'Pitch Motor Assembly', quantity: 1, cost: 85000 },
      { name: 'Pitch Bearing', quantity: 1, cost: 45000 },
      { name: 'Encoder Sensor', quantity: 1, cost: 12000 },
    ],
    laborHours: 24,
  },
  cyclic_vibration: {
    title: 'Cyclic Vibration / Blade Imbalance Detected',
    urgency: 'HIGH',
    actions: [
      'Perform rotor balancing procedure immediately',
      'Inspect affected blade for structural damage',
      'Check blade bolt torque specifications',
      'Analyze vibration spectrum for bearing wear',
    ],
    parts: [
      { name: 'Blade Balance Weight Kit', quantity: 1, cost: 15000 },
      { name: 'High-Strength Blade Bolts', quantity: 12, cost: 18000 },
      { name: 'Vibration Damper', quantity: 3, cost: 25000 },
    ],
    laborHours: 20,
  },
  normal: {
    title: 'Turbine Operating Normally',
    urgency: 'LOW',
    actions: [
      'Continue regular monitoring schedule',
      'No immediate action required',
      'Next scheduled maintenance as per OEM guidelines',
    ],
    parts: [],
    laborHours: 0,
  },
};

// Blade SVG Component
const BladeDiagram = ({ affectedBlade, severity }) => {
  const getBladeClass = (bladeNum) => {
    if (affectedBlade === bladeNum) {
      if (severity > 0.7) return 'blade-critical';
      if (severity > 0.4) return 'blade-warning';
      return 'blade-ok';
    }
    return 'blade-inactive';
  };

  return (
    <svg viewBox="0 0 200 200" className="w-full max-w-[240px] mx-auto">
      {/* Hub */}
      <circle cx="100" cy="100" r="15" fill="#334155" stroke="#475569" strokeWidth="2" />
      <circle cx="100" cy="100" r="8" fill="#1e293b" />
      
      {/* Blade 1 - Top */}
      <path
        d="M100 85 L95 20 Q100 10 105 20 L100 85"
        className={getBladeClass(1)}
        data-testid="blade-1"
      />
      
      {/* Blade 2 - Bottom Right */}
      <path
        d="M110 110 L160 160 Q170 155 165 145 L115 100"
        className={getBladeClass(2)}
        data-testid="blade-2"
      />
      
      {/* Blade 3 - Bottom Left */}
      <path
        d="M90 110 L40 160 Q30 155 35 145 L85 100"
        className={getBladeClass(3)}
        data-testid="blade-3"
      />

      {/* Blade Labels */}
      <text x="100" y="5" textAnchor="middle" className="text-xs fill-slate-500">B1</text>
      <text x="175" y="170" textAnchor="middle" className="text-xs fill-slate-500">B2</text>
      <text x="25" y="170" textAnchor="middle" className="text-xs fill-slate-500">B3</text>
      
      {/* Affected blade indicator */}
      {affectedBlade && (
        <text x="100" y="195" textAnchor="middle" className="text-xs fill-amber-500 font-bold">
          BLADE {affectedBlade} AFFECTED
        </text>
      )}
    </svg>
  );
};

// CSV Format Help Modal
const CSVFormatHelp = ({ onClose }) => (
  <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
    <div className="bg-slate-900 border border-slate-700 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-auto">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-100">Expected CSV Format</h3>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-200">✕</button>
      </div>
      <div className="p-4 space-y-4">
        <div>
          <h4 className="text-sm font-medium text-cyan-400 mb-2">Required Columns</h4>
          <div className="bg-slate-950 rounded p-3 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left p-2 text-slate-400">Column</th>
                  <th className="text-left p-2 text-slate-400">Description</th>
                  <th className="text-left p-2 text-slate-400">Example</th>
                </tr>
              </thead>
              <tbody>
                {CSV_FORMAT.required.map(col => (
                  <tr key={col.name} className="border-b border-slate-800/50">
                    <td className="p-2 text-cyan-300 font-mono">{col.name}</td>
                    <td className="p-2 text-slate-400">{col.desc}</td>
                    <td className="p-2 text-slate-500 font-mono">{col.example}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h4 className="text-sm font-medium text-amber-400 mb-2">Optional Columns</h4>
          <div className="bg-slate-950 rounded p-3">
            <table className="w-full text-xs">
              <tbody>
                {CSV_FORMAT.optional.map(col => (
                  <tr key={col.name} className="border-b border-slate-800/50">
                    <td className="p-2 text-amber-300 font-mono">{col.name}</td>
                    <td className="p-2 text-slate-400">{col.desc}</td>
                    <td className="p-2 text-slate-500 font-mono">{col.example}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="bg-slate-800/50 rounded p-3">
          <p className="text-xs text-slate-400">
            <strong className="text-slate-300">Note:</strong> Data should be sampled at 1Hz (1 second intervals).
            Minimum 300 rows (~5 minutes) required for accurate analysis. Upload 8+ hours for best results.
          </p>
        </div>
      </div>
    </div>
  </div>
);

export default function TurbineDetail() {
  const { id } = useParams();
  const [turbine, setTurbine] = useState(null);
  const [detections, setDetections] = useState([]);
  const [workOrders, setWorkOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [generatingWO, setGeneratingWO] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [showCSVHelp, setShowCSVHelp] = useState(false);
  const contentRef = useRef(null);

  useEffect(() => {
    fetchData();
  }, [id]);

  const fetchData = async () => {
    try {
      const [turbineRes, detectionsRes, workOrdersRes] = await Promise.all([
        axios.get(`${API}/turbines/${id}`),
        axios.get(`${API}/detections/${id}`),
        axios.get(`${API}/workorders/${id}`)
      ]);
      setTurbine(turbineRes.data);
      setDetections(detectionsRes.data);
      setWorkOrders(workOrdersRes.data);
    } catch (err) {
      toast.error('Failed to load turbine data');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
  };

  const analyzeCSV = async () => {
    if (!selectedFile) {
      toast.error('Please select a CSV file first');
      return;
    }

    setAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('turbine_id', id);

      const res = await axios.post(`${API}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      toast.success('Analysis complete!');
      setDetections([res.data, ...detections]);
      setSelectedFile(null);
      
      // Clear file input
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = '';
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const generateWorkOrder = async () => {
    const latestDetection = detections[0];
    if (!latestDetection) {
      toast.error('No detection to create work order from');
      return;
    }

    if (latestDetection.fault_type === 'normal') {
      toast.info('No work order needed - turbine is healthy');
      return;
    }

    const rec = RECOMMENDATIONS[latestDetection.fault_type] || RECOMMENDATIONS.normal;
    
    setGeneratingWO(true);
    try {
      const res = await axios.post(`${API}/workorders`, {
        detection_id: latestDetection.id,
        turbine_id: id,
        parts: rec.parts,
        labor_hours: rec.laborHours + (latestDetection.severity * 8),
        cost_estimate: rec.parts.reduce((sum, p) => sum + (p.cost * p.quantity), 0) + (rec.laborHours * 2500)
      });

      // Download PDF automatically
      if (res.data.pdf_base64) {
        const link = document.createElement('a');
        link.href = `data:application/pdf;base64,${res.data.pdf_base64}`;
        link.download = `WorkOrder_${turbine.name}_${new Date().toISOString().slice(0,10)}.pdf`;
        link.click();
      }

      toast.success('Work order generated and downloaded!');
      setWorkOrders([res.data, ...workOrders]);
    } catch (err) {
      toast.error('Failed to generate work order');
    } finally {
      setGeneratingWO(false);
    }
  };

  const downloadWorkOrder = (wo) => {
    if (wo.pdf_base64) {
      const link = document.createElement('a');
      link.href = `data:application/pdf;base64,${wo.pdf_base64}`;
      link.download = `WorkOrder_${wo.id.slice(0,8)}.pdf`;
      link.click();
      toast.success('PDF downloaded!');
    } else {
      toast.error('PDF not available');
    }
  };

  const exportScreenshot = async () => {
    if (!contentRef.current) return;
    
    try {
      const canvas = await html2canvas(contentRef.current, {
        backgroundColor: '#020617',
        scale: 2,
        width: 1920,
        height: 1080
      });
      
      const link = document.createElement('a');
      link.download = `${turbine?.name || 'turbine'}_diagnosis_${new Date().toISOString().slice(0,10)}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
      
      toast.success('Screenshot exported!');
    } catch (err) {
      toast.error('Failed to export screenshot');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#020617] flex items-center justify-center">
        <div className="text-cyan-500">Loading...</div>
      </div>
    );
  }

  if (!turbine) {
    return (
      <div className="min-h-screen bg-[#020617] flex items-center justify-center">
        <div className="text-rose-500">Turbine not found</div>
      </div>
    );
  }

  const latestDetection = detections[0];
  const mlResult = latestDetection?.ml_result || {};
  const forecast = mlResult.what_if_forecast || {};
  const recommendation = latestDetection ? RECOMMENDATIONS[latestDetection.fault_type] || RECOMMENDATIONS.normal : null;

  // Prepare chart data
  const residualData = (mlResult.residual_series || []).map((val, idx) => ({
    index: idx + 1,
    value: val
  }));

  const forecastData = [
    { name: 'If Fixed', energy: forecast.fix_energy_mwh || 0, fill: '#10b981' },
    { name: 'If Not Fixed', energy: forecast.nofix_energy_mwh || 0, fill: '#ef4444' }
  ];

  // Calculate ROI
  const roiData = latestDetection && latestDetection.fault_type !== 'normal' ? {
    revenueSaved: forecast.revenue_delta || 0,
    repairCost: recommendation?.parts.reduce((sum, p) => sum + (p.cost * p.quantity), 0) + (recommendation?.laborHours * 2500) || 0,
    netBenefit: (forecast.revenue_delta || 0) - (recommendation?.parts.reduce((sum, p) => sum + (p.cost * p.quantity), 0) + (recommendation?.laborHours * 2500) || 0),
    roi: ((forecast.revenue_delta || 0) / (recommendation?.parts.reduce((sum, p) => sum + (p.cost * p.quantity), 0) + (recommendation?.laborHours * 2500) || 1) * 100)
  } : null;

  return (
    <div className="min-h-screen bg-[#020617] grid-bg">
      {/* CSV Format Help Modal */}
      {showCSVHelp && <CSVFormatHelp onClose={() => setShowCSVHelp(false)} />}

      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/dashboard">
                <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200" data-testid="back-btn">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back
                </Button>
              </Link>
              <div>
                <h1 className="text-xl font-bold text-slate-100">{turbine.name}</h1>
                <div className="flex items-center gap-1 text-xs text-slate-500">
                  <MapPin className="w-3 h-3" />
                  {turbine.location}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={exportScreenshot}
                data-testid="export-screenshot-btn"
                className="border-slate-700 text-slate-300 hover:bg-slate-800"
              >
                <Download className="w-4 h-4 mr-2" />
                Screenshot
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main ref={contentRef} className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column */}
          <div className="lg:col-span-4 space-y-6">
            {/* Blade Diagram */}
            <Card className="bg-slate-900 border-slate-800" data-testid="blade-diagram-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-slate-200 flex items-center gap-2 text-base">
                  <Wind className="w-5 h-5 text-cyan-500" />
                  Blade Health
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col items-center">
                <BladeDiagram 
                  affectedBlade={latestDetection?.blade} 
                  severity={latestDetection?.severity || 0}
                />
                
                <div className="w-full mt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-500">Severity</span>
                    <span className="font-mono text-slate-300">{((latestDetection?.severity || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-3 bg-slate-800 rounded-sm overflow-hidden">
                    <div 
                      className="h-full severity-bar transition-all duration-500"
                      style={{ width: `${(latestDetection?.severity || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* CSV Upload */}
            <Card className="bg-slate-900 border-slate-800" data-testid="upload-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-slate-200 flex items-center justify-between text-base">
                  <span className="flex items-center gap-2">
                    <Upload className="w-5 h-5 text-cyan-500" />
                    Upload SCADA Data
                  </span>
                  <button 
                    onClick={() => setShowCSVHelp(true)}
                    className="text-cyan-500 hover:text-cyan-400"
                    title="View expected CSV format"
                  >
                    <Info className="w-4 h-4" />
                  </button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <Label className="text-slate-400 text-xs">Select CSV File</Label>
                  <Input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    data-testid="csv-upload-input"
                    className="bg-slate-950 border-slate-700 text-slate-300 file:bg-slate-800 file:text-slate-300 file:border-0 text-xs"
                  />
                </div>
                
                {selectedFile && (
                  <p className="text-xs text-slate-400">
                    Selected: {selectedFile.name}
                  </p>
                )}

                <Button
                  onClick={analyzeCSV}
                  disabled={!selectedFile || analyzing}
                  data-testid="analyze-btn"
                  className="w-full bg-cyan-600 hover:bg-cyan-500 text-white"
                >
                  <Activity className="w-4 h-4 mr-2" />
                  {analyzing ? 'Analyzing...' : 'Run ML Analysis'}
                </Button>

                <p className="text-xs text-slate-600">
                  Required: timestamp, wind_speed, rotor_rpm, power, pitch_cmd, pitch_meas, torque, vibration
                </p>
              </CardContent>
            </Card>

            {/* One-Click Work Order */}
            <Card className="bg-slate-900 border-slate-800" data-testid="workorder-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-slate-200 flex items-center gap-2 text-base">
                  <Wrench className="w-5 h-5 text-cyan-500" />
                  Maintenance Action
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button
                  onClick={generateWorkOrder}
                  disabled={!latestDetection || generatingWO || latestDetection?.fault_type === 'normal'}
                  data-testid="generate-workorder-btn"
                  className="w-full bg-orange-600 hover:bg-orange-500 text-white"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  {generatingWO ? 'Generating...' : 'Generate Work Order (One-Click)'}
                </Button>
                
                {latestDetection?.fault_type === 'normal' && (
                  <p className="text-xs text-emerald-500 text-center">
                    <CheckCircle className="w-3 h-3 inline mr-1" />
                    No maintenance required - turbine healthy
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Work Order History */}
            {workOrders.length > 0 && (
              <Card className="bg-slate-900 border-slate-800" data-testid="workorder-history-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-slate-200 text-base">Work Order History</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-[200px] overflow-y-auto">
                    {workOrders.map((wo) => (
                      <div 
                        key={wo.id}
                        className="flex items-center justify-between p-2 bg-slate-950 rounded border border-slate-800"
                      >
                        <div>
                          <p className="text-xs text-slate-300 font-mono">#{wo.id.slice(0,8)}</p>
                          <p className="text-xs text-slate-500">{new Date(wo.created_at).toLocaleDateString()}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            ₹{wo.cost_estimate?.toLocaleString()}
                          </Badge>
                          <Button 
                            size="sm" 
                            variant="ghost"
                            onClick={() => downloadWorkOrder(wo)}
                            className="h-7 w-7 p-0"
                          >
                            <FileDown className="w-3 h-3 text-cyan-500" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column */}
          <div className="lg:col-span-8 space-y-6">
            {/* Diagnosis Card */}
            <Card className={`border-2 ${
              !latestDetection ? 'bg-slate-900 border-slate-800' :
              latestDetection.fault_type === 'normal' ? 'bg-emerald-950/30 border-emerald-800' :
              latestDetection.severity > 0.7 ? 'bg-rose-950/30 border-rose-800' :
              'bg-amber-950/30 border-amber-800'
            }`} data-testid="diagnosis-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-slate-100 flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Gauge className="w-5 h-5 text-cyan-500" />
                    Diagnosis Card
                  </span>
                  {latestDetection && (
                    <Badge 
                      className={`${
                        latestDetection.fault_type === 'normal' ? 'bg-emerald-600' :
                        latestDetection.severity > 0.7 ? 'bg-rose-600' : 'bg-amber-600'
                      } text-white`}
                    >
                      {recommendation?.urgency} PRIORITY
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {latestDetection ? (
                  <div className="space-y-4">
                    {/* Main diagnosis info */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-slate-900/50 p-3 rounded border border-slate-800">
                        <p className="metric-label">Fault Type</p>
                        <p className="metric-value text-lg text-slate-100 capitalize">
                          {latestDetection.fault_type?.replace('_', ' ')}
                        </p>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded border border-slate-800">
                        <p className="metric-label">Affected Blade</p>
                        <p className="metric-value text-lg text-slate-100">
                          {latestDetection.blade ? `Blade ${latestDetection.blade}` : 'N/A'}
                        </p>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded border border-slate-800">
                        <p className="metric-label">Severity</p>
                        <p className={`metric-value text-lg ${
                          latestDetection.severity > 0.7 ? 'text-rose-400' :
                          latestDetection.severity > 0.4 ? 'text-amber-400' : 'text-emerald-400'
                        }`}>
                          {(latestDetection.severity * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded border border-slate-800">
                        <p className="metric-label">Confidence</p>
                        <p className="metric-value text-lg text-cyan-400">
                          {(latestDetection.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>

                    {/* Recommendation title */}
                    <div className={`p-3 rounded border ${
                      latestDetection.fault_type === 'normal' ? 'bg-emerald-900/20 border-emerald-800' :
                      'bg-amber-900/20 border-amber-800'
                    }`}>
                      <p className={`font-semibold ${
                        latestDetection.fault_type === 'normal' ? 'text-emerald-400' : 'text-amber-400'
                      }`}>
                        {recommendation?.title}
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-500 text-center py-8">
                    Upload SCADA CSV data to generate diagnosis
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Recommendations & Actions */}
            {latestDetection && latestDetection.fault_type !== 'normal' && (
              <Card className="bg-slate-900 border-slate-800" data-testid="recommendations-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-slate-200 flex items-center gap-2">
                    <Wrench className="w-5 h-5 text-cyan-500" />
                    Remediation Recommendations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Action items */}
                    <div>
                      <p className="text-xs text-slate-500 uppercase mb-2">Recommended Actions</p>
                      <div className="space-y-2">
                        {recommendation?.actions.map((action, idx) => (
                          <div key={idx} className="flex items-start gap-2 text-sm">
                            <span className="text-cyan-500 font-mono">{idx + 1}.</span>
                            <span className="text-slate-300">{action}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Required parts */}
                    {recommendation?.parts.length > 0 && (
                      <div>
                        <p className="text-xs text-slate-500 uppercase mb-2">Required Parts & Estimated Costs</p>
                        <div className="bg-slate-950 rounded border border-slate-800 overflow-hidden">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b border-slate-800 bg-slate-900">
                                <th className="text-left p-2 text-slate-400">Part</th>
                                <th className="text-center p-2 text-slate-400">Qty</th>
                                <th className="text-right p-2 text-slate-400">Cost (₹)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {recommendation.parts.map((part, idx) => (
                                <tr key={idx} className="border-b border-slate-800/50">
                                  <td className="p-2 text-slate-300">{part.name}</td>
                                  <td className="p-2 text-center text-slate-400">{part.quantity}</td>
                                  <td className="p-2 text-right text-slate-300 font-mono">
                                    {(part.cost * part.quantity).toLocaleString()}
                                  </td>
                                </tr>
                              ))}
                              <tr className="bg-slate-900">
                                <td className="p-2 text-slate-300 font-medium">Labor ({recommendation.laborHours}h)</td>
                                <td className="p-2 text-center text-slate-400">-</td>
                                <td className="p-2 text-right text-slate-300 font-mono">
                                  {(recommendation.laborHours * 2500).toLocaleString()}
                                </td>
                              </tr>
                              <tr className="bg-cyan-900/20">
                                <td colSpan={2} className="p-2 text-cyan-300 font-semibold">Total Estimated Cost</td>
                                <td className="p-2 text-right text-cyan-300 font-mono font-semibold">
                                  ₹{(recommendation.parts.reduce((sum, p) => sum + (p.cost * p.quantity), 0) + (recommendation.laborHours * 2500)).toLocaleString()}
                                </td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* ROI Estimation */}
            {roiData && (
              <Card className="bg-slate-900 border-slate-800" data-testid="roi-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-slate-200 flex items-center gap-2">
                    <DollarSign className="w-5 h-5 text-cyan-500" />
                    ROI Estimation (Monthly)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-slate-950 p-3 rounded border border-slate-800">
                      <p className="metric-label">Revenue Saved</p>
                      <p className="metric-value text-xl text-emerald-400">
                        ₹{roiData.revenueSaved.toLocaleString()}
                      </p>
                    </div>
                    <div className="bg-slate-950 p-3 rounded border border-slate-800">
                      <p className="metric-label">Repair Cost</p>
                      <p className="metric-value text-xl text-amber-400">
                        ₹{roiData.repairCost.toLocaleString()}
                      </p>
                    </div>
                    <div className="bg-slate-950 p-3 rounded border border-slate-800">
                      <p className="metric-label">Net Benefit</p>
                      <p className={`metric-value text-xl ${roiData.netBenefit > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        ₹{roiData.netBenefit.toLocaleString()}
                      </p>
                    </div>
                    <div className="bg-cyan-900/30 p-3 rounded border border-cyan-800">
                      <p className="metric-label">ROI</p>
                      <p className="metric-value text-xl text-cyan-400">
                        {roiData.roi.toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* What-If Forecast Chart */}
            {forecast.fix_energy_mwh && (
              <Card className="bg-slate-900 border-slate-800" data-testid="forecast-chart-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-slate-200 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-cyan-500" />
                    What-If Energy Forecast (Monthly)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[180px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={forecastData} layout="vertical">
                        <XAxis type="number" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                        <YAxis dataKey="name" type="category" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} width={100} />
                        <Tooltip 
                          contentStyle={{ 
                            background: 'rgba(15, 23, 42, 0.95)', 
                            border: '1px solid #334155',
                            borderRadius: '4px'
                          }}
                          formatter={(value) => [`${value} MWh`, 'Energy']}
                        />
                        <Bar dataKey="energy" radius={[0, 4, 4, 0]}>
                          {forecastData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-4 p-3 bg-emerald-900/20 rounded border border-emerald-800">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-emerald-500" />
                        <span className="text-slate-300">Revenue Impact if Repaired</span>
                      </div>
                      <span className="metric-value text-xl text-emerald-400">
                        +₹{forecast.revenue_delta?.toLocaleString() || 0}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Residual Chart */}
            {residualData.length > 0 && (
              <Card className="bg-slate-900 border-slate-800" data-testid="residual-chart-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-slate-200 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-cyan-500" />
                    Power Residual Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[150px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={residualData}>
                        <XAxis 
                          dataKey="index" 
                          stroke="#64748b" 
                          tick={{ fill: '#64748b', fontSize: 12 }}
                        />
                        <YAxis 
                          stroke="#64748b" 
                          tick={{ fill: '#64748b', fontSize: 12 }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            background: 'rgba(15, 23, 42, 0.95)', 
                            border: '1px solid #334155',
                            borderRadius: '4px'
                          }}
                          labelStyle={{ color: '#94a3b8' }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#06b6d4" 
                          strokeWidth={2}
                          dot={{ fill: '#06b6d4', r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
