import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "sonner";
import { AuthProvider, useAuth } from "./context/AuthContext";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import TurbineDetail from "./pages/TurbineDetail";
import Simulator from "./pages/Simulator";
import Admin from "./pages/Admin";

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen bg-[#020617] flex items-center justify-center">
        <div className="text-cyan-500 text-xl">Loading...</div>
      </div>
    );
  }
  
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
};

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <div className="min-h-screen bg-[#020617]">
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/dashboard" element={
              <ProtectedRoute><Dashboard /></ProtectedRoute>
            } />
            <Route path="/turbines/:id" element={
              <ProtectedRoute><TurbineDetail /></ProtectedRoute>
            } />
            <Route path="/simulator" element={
              <ProtectedRoute><Simulator /></ProtectedRoute>
            } />
            <Route path="/admin" element={
              <ProtectedRoute><Admin /></ProtectedRoute>
            } />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Routes>
          <Toaster 
            position="top-right" 
            toastOptions={{
              style: {
                background: '#0f172a',
                color: '#f8fafc',
                border: '1px solid #334155'
              }
            }}
          />
        </div>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
