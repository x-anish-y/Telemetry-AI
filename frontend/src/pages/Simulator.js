import { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Input } from '../components/ui/input';
import { Textarea } from '../components/ui/textarea';
import { ArrowLeft, Wand2, Download, Database, FileSpreadsheet, Info } from 'lucide-react';
import { toast } from 'sonner';

const API = process.env.REACT_APP_BACKEND_URL
  ? `${process.env.REACT_APP_BACKEND_URL}/api`
  : '/api';

// Expected output columns
const OUTPUT_COLUMNS = [
  { name: 'timestamp', desc: 'ISO datetime (1Hz sampling)' },
  { name: 'wind_speed', desc: 'Wind velocity (m/s)' },
  { name: 'rotor_rpm', desc: 'Rotor speed (RPM)' },
  { name: 'power', desc: 'Active power (kW)' },
  { name: 'pitch_cmd', desc: 'Commanded pitch (°)' },
  { name: 'pitch_meas', desc: 'Measured pitch (°)' },
  { name: 'torque', desc: 'Generator torque (kNm)' },
  { name: 'vibration', desc: 'Nacelle vibration (m/s²)' },
  { name: 'temperature', desc: 'Gearbox temp (°C)' },
];

export default function Simulator() {
  const [faultType, setFaultType] = useState('normal');
  const [rows, setRows] = useState(500);
  const [generating, setGenerating] = useState(false);
  const [csvContent, setCsvContent] = useState('');
  const [csvPreview, setCsvPreview] = useState([]);

  const generateData = async () => {
    setGenerating(true);
    try {
      const formData = new FormData();
      formData.append('fault_type', faultType);
      formData.append('rows', rows);

      const res = await axios.post(`${API}/simulate`, formData);
      setCsvContent(res.data.csv_content);
      
      // Parse preview
      const lines = res.data.csv_content.split('\n').slice(0, 11);
      setCsvPreview(lines);
      
      toast.success(`Generated ${rows} rows of ${faultType} data`);
    } catch (err) {
      toast.error('Failed to generate data');
    } finally {
      setGenerating(false);
    }
  };

  const downloadCSV = () => {
    if (!csvContent) return;
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `scada_${faultType}_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    
    toast.success('CSV downloaded!');
  };

  return (
    <div className="min-h-screen bg-[#020617] grid-bg">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center gap-4">
            <Link to="/dashboard">
              <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200" data-testid="back-btn">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
            </Link>
            <div>
              <h1 className="text-xl font-bold text-slate-100">SCADA Simulator</h1>
              <p className="text-xs text-slate-500">Generate synthetic telemetry data for testing</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Generator Settings */}
          <Card className="bg-slate-900 border-slate-800" data-testid="generator-card">
            <CardHeader>
              <CardTitle className="text-slate-200 flex items-center gap-2">
                <Wand2 className="w-5 h-5 text-cyan-500" />
                Data Generator
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label className="text-slate-400">Fault Type</Label>
                <Select value={faultType} onValueChange={setFaultType}>
                  <SelectTrigger data-testid="fault-type-select" className="bg-slate-950 border-slate-700 text-slate-300">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-900 border-slate-700">
                    <SelectItem value="normal">Normal (Healthy)</SelectItem>
                    <SelectItem value="erosion">Blade Erosion</SelectItem>
                    <SelectItem value="cyclic_vibration">Cyclic Vibration</SelectItem>
                    <SelectItem value="pitch_drift">Pitch Drift</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-500">
                  {faultType === 'normal' && 'Generate baseline healthy turbine data'}
                  {faultType === 'erosion' && 'Simulates gradual power decline from blade surface erosion'}
                  {faultType === 'cyclic_vibration' && 'Adds periodic vibration spikes indicating imbalance'}
                  {faultType === 'pitch_drift' && 'Introduces high variance in pitch angle readings'}
                </p>
              </div>

              <div className="space-y-2">
                <Label className="text-slate-400">Number of Rows</Label>
                <Input
                  type="number"
                  min="100"
                  max="3600"
                  value={rows}
                  onChange={(e) => setRows(parseInt(e.target.value) || 500)}
                  data-testid="rows-input"
                  className="bg-slate-950 border-slate-700 text-slate-300"
                />
                <p className="text-xs text-slate-500">1Hz sampling (500 rows = ~8 minutes, 3600 = 1 hour)</p>
              </div>

              <Button
                onClick={generateData}
                disabled={generating}
                data-testid="generate-btn"
                className="w-full bg-cyan-600 hover:bg-cyan-500 text-white"
              >
                <Database className="w-4 h-4 mr-2" />
                {generating ? 'Generating...' : 'Generate Data'}
              </Button>

              {/* Output columns info */}
              <div className="pt-3 border-t border-slate-800">
                <p className="text-xs text-slate-500 mb-2 flex items-center gap-1">
                  <Info className="w-3 h-3" /> Output Columns:
                </p>
                <div className="grid grid-cols-3 gap-1 text-xs">
                  {OUTPUT_COLUMNS.map(col => (
                    <span key={col.name} className="text-cyan-400 font-mono">{col.name}</span>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* CSV Preview */}
          <Card className="bg-slate-900 border-slate-800" data-testid="preview-card">
            <CardHeader>
              <CardTitle className="text-slate-200 flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <FileSpreadsheet className="w-5 h-5 text-cyan-500" />
                  Preview
                </span>
                {csvContent && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={downloadCSV}
                    data-testid="download-csv-btn"
                    className="border-slate-700 text-slate-300 hover:bg-slate-800"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download CSV
                  </Button>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {csvPreview.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="border-b border-slate-800">
                        {csvPreview[0].split(',').map((header, i) => (
                          <th key={i} className="text-left p-2 text-slate-500 uppercase">
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {csvPreview.slice(1).map((row, i) => (
                        <tr key={i} className="border-b border-slate-800/50">
                          {row.split(',').map((cell, j) => (
                            <td key={j} className="p-2 text-slate-400">
                              {cell.length > 16 ? cell.slice(0, 16) + '...' : cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-600">
                  <FileSpreadsheet className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Generate data to see preview</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Raw CSV */}
        {csvContent && (
          <Card className="bg-slate-900 border-slate-800 mt-6" data-testid="raw-csv-card">
            <CardHeader>
              <CardTitle className="text-slate-200">Raw CSV Content</CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                value={csvContent}
                readOnly
                className="font-mono text-xs bg-slate-950 border-slate-700 text-slate-400 h-48"
              />
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
