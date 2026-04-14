import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Wind, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

export default function Login() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        await login(email, password);
        toast.success('Welcome back!');
      } else {
        await register(email, password, name);
        toast.success('Account created successfully!');
      }
      navigate('/dashboard');
    } catch (err) {
      const msg = err.response?.data?.detail || 'Authentication failed';
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen grid-bg flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center opacity-20"
        style={{ 
          backgroundImage: 'url(https://images.unsplash.com/photo-1579297658873-f49f379d28ad?crop=entropy&cs=srgb&fm=jpg&q=85)'
        }}
      />
      <div className="absolute inset-0 bg-gradient-to-b from-[#020617]/80 via-[#020617]/90 to-[#020617]" />

      <Card className="w-full max-w-md bg-slate-900/80 backdrop-blur-xl border-slate-800 relative z-10" data-testid="login-card">
        <CardHeader className="text-center space-y-4">
          <div className="mx-auto w-16 h-16 bg-cyan-500/10 rounded-sm flex items-center justify-center border border-cyan-500/30">
            <Wind className="w-8 h-8 text-cyan-500" />
          </div>
          <CardTitle className="text-2xl font-bold tracking-tight text-slate-100">
            Telemetry AI
          </CardTitle>
          <p className="text-slate-400 text-sm">
            AI-Powered Predictive Maintenance System
          </p>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div className="space-y-2">
                <Label htmlFor="name" className="text-slate-300">Name</Label>
                <Input
                  id="name"
                  data-testid="name-input"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Enter your name"
                  className="bg-slate-950 border-slate-700 text-slate-100 focus:border-cyan-500"
                  required={!isLogin}
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email" className="text-slate-300">Email</Label>
              <Input
                id="email"
                data-testid="email-input"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="operator@windfarm.com"
                className="bg-slate-950 border-slate-700 text-slate-100 focus:border-cyan-500"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-slate-300">Password</Label>
              <Input
                id="password"
                data-testid="password-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="bg-slate-950 border-slate-700 text-slate-100 focus:border-cyan-500"
                required
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded-sm text-red-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            )}

            <Button
              type="submit"
              data-testid="submit-btn"
              disabled={loading}
              className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold tracking-wide shadow-[0_0_10px_rgba(8,145,178,0.5)] rounded-sm"
            >
              {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <button
              type="button"
              data-testid="toggle-auth-mode"
              onClick={() => {
                setIsLogin(!isLogin);
                setError('');
              }}
              className="text-cyan-500 hover:text-cyan-400 text-sm transition-colors"
            >
              {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
