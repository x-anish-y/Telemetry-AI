#!/usr/bin/env python3
"""
ML Integration Testing for Wind Turbine Predictive Maintenance System
Tests the real ML engine integration with trained models.
"""

import requests
import sys
import json
import os
from datetime import datetime
from pathlib import Path

class MLIntegrationTester:
    def __init__(self, base_url="https://windml-predictor.preview.emergentagent.com"):
        self.base_url = base_url
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {'Authorization': f'Bearer {self.token}'} if self.token else {}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
                else:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'PUT':
                headers['Content-Type'] = 'application/json'
                response = requests.put(url, json=data, headers=headers, timeout=30)

            success = response.status_code == expected_status
            response_data = {}
            
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text[:200]}

            details = f"Status: {response.status_code}"
            if not success:
                details += f", Response: {response_data}"

            self.log_test(name, success, details)
            return success, response_data

        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_login(self):
        """Test login with provided credentials"""
        success, response = self.run_test(
            "Login with ML test credentials",
            "POST",
            "auth/login",
            200,
            data={"email": "mltest@windfarm.com", "password": "test123456"}
        )
        if success and 'access_token' in response:
            self.token = response['access_token']
            return True
        return False

    def test_ml_status(self):
        """Test ML status endpoint"""
        success, response = self.run_test(
            "GET /api/ml-status - Check ML engine status",
            "GET",
            "ml-status",
            200
        )
        
        if success:
            # Verify response structure
            required_fields = ['status', 'models_loaded', 'feature_importance']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                self.log_test(
                    "ML Status Response Structure",
                    False,
                    f"Missing fields: {missing_fields}"
                )
                return False
            
            # Check if models are loaded
            if response['status'] != 'ready':
                self.log_test(
                    "ML Models Status",
                    False,
                    f"Status: {response['status']}, Expected: ready"
                )
                return False
            
            # Verify expected models
            expected_models = [
                "anomaly_detector (Isolation Forest)",
                "fault_classifier (Random Forest)", 
                "severity_estimator (RF Regressor)",
                "whatif_forecaster (Gradient Boosting)"
            ]
            
            loaded_models = response.get('models_loaded', [])
            missing_models = [model for model in expected_models if model not in loaded_models]
            
            if missing_models:
                self.log_test(
                    "ML Models Loaded Check",
                    False,
                    f"Missing models: {missing_models}"
                )
                return False
            
            # Check feature importance structure
            feature_importance = response.get('feature_importance', {})
            if 'features' not in feature_importance or 'importance' not in feature_importance:
                self.log_test(
                    "Feature Importance Structure",
                    False,
                    "Missing features or importance arrays"
                )
                return False
            
            self.log_test("ML Status Response Structure", True)
            self.log_test("ML Models Status", True)
            self.log_test("ML Models Loaded Check", True)
            self.log_test("Feature Importance Structure", True)
            
            print(f"   📊 Loaded Models: {len(loaded_models)}")
            print(f"   🧠 Top Features: {feature_importance['features'][:3]}")
            
        return success

    def test_healthy_data_analysis(self):
        """Test ML analysis with healthy data"""
        # Load healthy CSV data
        healthy_csv_path = "/app/turbine_ml/examples/healthy.csv"
        
        if not os.path.exists(healthy_csv_path):
            self.log_test(
                "Load Healthy CSV Data",
                False,
                f"File not found: {healthy_csv_path}"
            )
            return False
        
        # Get a turbine ID first
        success, turbines_response = self.run_test(
            "Get Turbines for Analysis",
            "GET",
            "turbines",
            200
        )
        
        if not success or not turbines_response:
            self.log_test(
                "Get Turbine ID for Analysis",
                False,
                "No turbines available"
            )
            return False
        
        turbine_id = turbines_response[0]['id']
        
        # Upload healthy CSV
        with open(healthy_csv_path, 'rb') as f:
            files = {'file': ('healthy.csv', f, 'text/csv')}
            data = {'turbine_id': turbine_id}
            
            success, response = self.run_test(
                "POST /api/analyze - Healthy Data Analysis",
                "POST",
                "analyze",
                200,
                data=data,
                files=files
            )
        
        if success:
            # Verify response structure for healthy data
            required_fields = ['fault_type', 'blade', 'severity', 'confidence', 'ml_result']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                self.log_test(
                    "Healthy Data Response Structure",
                    False,
                    f"Missing fields: {missing_fields}"
                )
                return False
            
            # Check if healthy data is correctly identified as 'normal'
            if response['fault_type'] != 'normal':
                self.log_test(
                    "Healthy Data Classification",
                    False,
                    f"Expected 'normal', got '{response['fault_type']}'"
                )
                return False
            
            self.log_test("Healthy Data Response Structure", True)
            self.log_test("Healthy Data Classification", True)
            
            print(f"   🟢 Fault Type: {response['fault_type']}")
            print(f"   📊 Confidence: {response['confidence']:.2f}")
            print(f"   ⚡ Severity: {response['severity']:.2f}")
            
        return success

    def test_faulty_data_analysis(self):
        """Test ML analysis with faulty data"""
        # Check if faulty demo file exists
        faulty_csv_path = "/app/turbine_ml/examples/demo_test.csv"
        
        if not os.path.exists(faulty_csv_path):
            # Use the regular demo_test.csv
            faulty_csv_path = "/app/turbine_ml/examples/demo_test.csv"
        
        if not os.path.exists(faulty_csv_path):
            self.log_test(
                "Load Faulty CSV Data",
                False,
                f"File not found: {faulty_csv_path}"
            )
            return False
        
        # Get a turbine ID
        success, turbines_response = self.run_test(
            "Get Turbines for Faulty Analysis",
            "GET",
            "turbines",
            200
        )
        
        if not success or not turbines_response:
            return False
        
        turbine_id = turbines_response[0]['id']
        
        # Upload faulty CSV
        with open(faulty_csv_path, 'rb') as f:
            files = {'file': ('demo_test.csv', f, 'text/csv')}
            data = {'turbine_id': turbine_id}
            
            success, response = self.run_test(
                "POST /api/analyze - Faulty Data Analysis",
                "POST",
                "analyze",
                200,
                data=data,
                files=files
            )
        
        if success:
            # Verify response structure
            required_fields = ['fault_type', 'blade', 'severity', 'confidence', 'ml_result']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                self.log_test(
                    "Faulty Data Response Structure",
                    False,
                    f"Missing fields: {missing_fields}"
                )
                return False
            
            # Check if faulty data is correctly identified (not 'normal')
            if response['fault_type'] == 'normal':
                self.log_test(
                    "Faulty Data Classification",
                    False,
                    "Faulty data incorrectly classified as 'normal'"
                )
                return False
            
            # Check if blade attribution is provided for faults
            if response['fault_type'] != 'normal' and response['blade'] is None:
                self.log_test(
                    "Blade Attribution for Faults",
                    False,
                    "No blade attribution provided for detected fault"
                )
                return False
            
            self.log_test("Faulty Data Response Structure", True)
            self.log_test("Faulty Data Classification", True)
            self.log_test("Blade Attribution for Faults", True)
            
            print(f"   🔴 Fault Type: {response['fault_type']}")
            print(f"   🔧 Affected Blade: {response['blade']}")
            print(f"   📊 Confidence: {response['confidence']:.2f}")
            print(f"   ⚡ Severity: {response['severity']:.2f}")
            
        return success

    def test_ml_result_structure(self):
        """Test that ML result contains all required fields"""
        # Use demo_test.csv for this test
        demo_csv_path = "/app/turbine_ml/examples/demo_test.csv"
        
        if not os.path.exists(demo_csv_path):
            self.log_test(
                "Load Demo CSV for ML Result Test",
                False,
                f"File not found: {demo_csv_path}"
            )
            return False
        
        # Get turbine ID
        success, turbines_response = self.run_test(
            "Get Turbines for ML Result Test",
            "GET",
            "turbines",
            200
        )
        
        if not success or not turbines_response:
            return False
        
        turbine_id = turbines_response[0]['id']
        
        # Upload CSV and analyze
        with open(demo_csv_path, 'rb') as f:
            files = {'file': ('demo_test.csv', f, 'text/csv')}
            data = {'turbine_id': turbine_id}
            
            success, response = self.run_test(
                "ML Result Structure Test",
                "POST",
                "analyze",
                200,
                data=data,
                files=files
            )
        
        if success:
            ml_result = response.get('ml_result', {})
            
            # Check required ML result fields
            required_ml_fields = [
                'detected', 'anomaly_count', 'fault_type', 'severity', 
                'confidence', 'residual_series', 'what_if_forecast'
            ]
            
            missing_ml_fields = [field for field in required_ml_fields if field not in ml_result]
            
            if missing_ml_fields:
                self.log_test(
                    "ML Result Fields Check",
                    False,
                    f"Missing ML result fields: {missing_ml_fields}"
                )
                return False
            
            # Check what_if_forecast structure
            forecast = ml_result.get('what_if_forecast', {})
            required_forecast_fields = ['fix_energy_mwh', 'nofix_energy_mwh', 'revenue_delta']
            missing_forecast_fields = [field for field in required_forecast_fields if field not in forecast]
            
            if missing_forecast_fields:
                self.log_test(
                    "What-If Forecast Structure",
                    False,
                    f"Missing forecast fields: {missing_forecast_fields}"
                )
                return False
            
            self.log_test("ML Result Fields Check", True)
            self.log_test("What-If Forecast Structure", True)
            
            print(f"   🔍 Anomalies Detected: {ml_result['anomaly_count']}")
            print(f"   📈 Revenue Impact: ₹{forecast.get('revenue_delta', 0):,.0f}")
            
        return success

    def run_all_tests(self):
        """Run all ML integration tests"""
        print("🧪 Starting ML Integration Tests for Wind Turbine Predictive Maintenance")
        print("=" * 80)
        
        # Login first
        if not self.test_login():
            print("❌ Login failed, stopping tests")
            return False
        
        # Test ML status endpoint
        print("\n📊 Testing ML Engine Status...")
        self.test_ml_status()
        
        # Test healthy data analysis
        print("\n🟢 Testing Healthy Data Analysis...")
        self.test_healthy_data_analysis()
        
        # Test faulty data analysis  
        print("\n🔴 Testing Faulty Data Analysis...")
        self.test_faulty_data_analysis()
        
        # Test ML result structure
        print("\n🔬 Testing ML Result Structure...")
        self.test_ml_result_structure()
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"📋 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        print(f"✅ Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        return self.tests_passed == self.tests_run

def main():
    tester = MLIntegrationTester()
    success = tester.run_all_tests()
    
    # Save detailed results
    results = {
        "summary": {
            "total_tests": tester.tests_run,
            "passed_tests": tester.tests_passed,
            "success_rate": f"{(tester.tests_passed/tester.tests_run)*100:.1f}%",
            "timestamp": datetime.now().isoformat()
        },
        "test_results": tester.test_results
    }
    
    with open("/app/ml_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())