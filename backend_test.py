import requests
import sys
import json
import time
from datetime import datetime
from io import StringIO

class TurbineFaultDetectAPITester:
    def __init__(self, base_url="https://windml-predictor.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.user_id = None
        self.turbine_id = None
        self.detection_id = None

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if details:
            print(f"    {details}")

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, form_data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        print(f"\n🔍 Testing {name}...")
        
        try:
            if form_data:
                # For form data, don't set Content-Type header
                headers.pop('Content-Type', None)
                if method == 'POST':
                    response = requests.post(url, data=form_data, files=files, headers=headers)
                else:
                    response = requests.request(method, url, data=form_data, files=files, headers=headers)
            elif method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}"
            
            if not success:
                details += f" (Expected {expected_status})"
                try:
                    error_data = response.json()
                    details += f" - {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f" - {response.text[:100]}"
            
            self.log_test(name, success, details)
            
            if success:
                try:
                    return response.json()
                except:
                    return {"status": "success"}
            return {}

        except Exception as e:
            self.log_test(name, False, f"Error: {str(e)}")
            return {}

    def test_health_check(self):
        """Test health endpoint"""
        response = self.run_test(
            "Health Check",
            "GET",
            "health",
            200
        )
        return response.get('status') == 'healthy'

    def test_user_registration(self):
        """Test user registration"""
        test_user = {
            "email": "test@windfarm.com",
            "password": "test123456",
            "name": "Test Operator"
        }
        
        response = self.run_test(
            "User Registration",
            "POST",
            "auth/register",
            200,
            data=test_user
        )
        
        if response and 'access_token' in response:
            self.token = response['access_token']
            self.user_id = response['user']['id']
            return True
        return False

    def test_user_login(self):
        """Test user login"""
        login_data = {
            "email": "test@windfarm.com",
            "password": "test123456"
        }
        
        response = self.run_test(
            "User Login",
            "POST",
            "auth/login",
            200,
            data=login_data
        )
        
        if response and 'access_token' in response:
            self.token = response['access_token']
            self.user_id = response['user']['id']
            return True
        return False

    def test_protected_me_endpoint(self):
        """Test protected /auth/me endpoint"""
        if not self.token:
            self.log_test("Protected /auth/me", False, "No token available")
            return False
            
        response = self.run_test(
            "Protected /auth/me",
            "GET",
            "auth/me",
            200
        )
        return 'id' in response

    def test_seed_demo_turbines(self):
        """Test seeding demo turbines"""
        response = self.run_test(
            "Seed Demo Turbines",
            "POST",
            "seed",
            200
        )
        return 'created' in response

    def test_list_turbines(self):
        """Test listing turbines"""
        response = self.run_test(
            "List Turbines",
            "GET",
            "turbines",
            200
        )
        
        if isinstance(response, list) and len(response) > 0:
            self.turbine_id = response[0]['id']
            return True
        return False

    def test_get_single_turbine(self):
        """Test getting single turbine"""
        if not self.turbine_id:
            self.log_test("Get Single Turbine", False, "No turbine ID available")
            return False
            
        response = self.run_test(
            "Get Single Turbine",
            "GET",
            f"turbines/{self.turbine_id}",
            200
        )
        return 'id' in response

    def test_simulate_scada_data(self):
        """Test SCADA data simulation"""
        form_data = {
            'fault_type': 'cyclic_vibration',
            'rows': '50'
        }
        
        response = self.run_test(
            "Simulate SCADA Data",
            "POST",
            "simulate",
            200,
            form_data=form_data
        )
        return 'csv_content' in response

    def test_analyze_csv(self):
        """Test CSV analysis with file upload"""
        if not self.turbine_id:
            self.log_test("Analyze CSV", False, "No turbine ID available")
            return False
        
        # First generate some test CSV data
        csv_content = """timestamp,turbine_id,wind_speed,rotor_speed,power_output,temperature,pitch_angle,vibration
2024-01-01 00:00:00,T001,12.5,15.0,1250.5,35.2,5.1,2.1
2024-01-01 00:10:00,T001,13.2,15.8,1380.2,35.8,5.3,2.3
2024-01-01 00:20:00,T001,11.8,14.2,1120.8,34.9,4.9,2.0
2024-01-01 00:30:00,T001,14.1,16.9,1520.3,36.5,5.5,2.5
2024-01-01 00:40:00,T001,12.9,15.5,1310.7,35.4,5.2,2.2"""
        
        files = {'file': ('test_data.csv', csv_content, 'text/csv')}
        form_data = {'turbine_id': self.turbine_id}
        
        response = self.run_test(
            "Analyze CSV",
            "POST",
            "analyze",
            200,
            form_data=form_data,
            files=files
        )
        
        if response and 'id' in response:
            self.detection_id = response['id']
            return True
        return False

    def test_get_detections(self):
        """Test getting detections for a turbine"""
        if not self.turbine_id:
            self.log_test("Get Detections", False, "No turbine ID available")
            return False
            
        response = self.run_test(
            "Get Detections",
            "GET",
            f"detections/{self.turbine_id}",
            200
        )
        return isinstance(response, list)

    def test_get_single_detection(self):
        """Test getting single detection"""
        if not self.detection_id:
            self.log_test("Get Single Detection", False, "No detection ID available")
            return False
            
        response = self.run_test(
            "Get Single Detection",
            "GET",
            f"detection/{self.detection_id}",
            200
        )
        return 'id' in response

    def test_generate_work_order(self):
        """Test work order generation"""
        if not self.detection_id or not self.turbine_id:
            self.log_test("Generate Work Order", False, "Missing detection or turbine ID")
            return False
        
        work_order_data = {
            "detection_id": self.detection_id,
            "turbine_id": self.turbine_id,
            "parts": [
                {"name": "Blade Repair Kit", "quantity": 1, "cost": 25000},
                {"name": "Pitch Motor", "quantity": 1, "cost": 45000}
            ],
            "labor_hours": 8.0,
            "cost_estimate": 75000.0
        }
        
        response = self.run_test(
            "Generate Work Order PDF",
            "POST",
            "workorders",
            200,
            data=work_order_data
        )
        return 'pdf_base64' in response

    def test_get_work_orders(self):
        """Test getting work orders for a turbine"""
        if not self.turbine_id:
            self.log_test("Get Work Orders", False, "No turbine ID available")
            return False
            
        response = self.run_test(
            "Get Work Orders",
            "GET",
            f"workorders/{self.turbine_id}",
            200
        )
        return isinstance(response, list)

    def test_get_config(self):
        """Test getting system config"""
        response = self.run_test(
            "Get Config",
            "GET",
            "config",
            200
        )
        return 'demo_mode' in response

    def test_update_config(self):
        """Test updating system config"""
        config_data = {"demo_mode": True}
        
        response = self.run_test(
            "Update Config",
            "PUT",
            "config",
            200,
            data=config_data
        )
        return 'demo_mode' in response

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting Turbine Fault Detect API Tests")
        print(f"📡 Base URL: {self.base_url}")
        print("=" * 60)
        
        # Health check first
        if not self.test_health_check():
            print("❌ Health check failed - stopping tests")
            return False
        
        # Authentication tests
        if not self.test_user_registration():
            # Try login if registration fails (user might already exist)
            if not self.test_user_login():
                print("❌ Authentication failed - stopping tests")
                return False
        
        self.test_protected_me_endpoint()
        
        # Turbine management tests
        self.test_seed_demo_turbines()
        self.test_list_turbines()
        self.test_get_single_turbine()
        
        # SCADA and analysis tests
        self.test_simulate_scada_data()
        self.test_analyze_csv()
        self.test_get_detections()
        self.test_get_single_detection()
        
        # Work order tests
        self.test_generate_work_order()
        self.test_get_work_orders()
        
        # Config tests
        self.test_get_config()
        self.test_update_config()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Tests completed: {self.tests_passed}/{self.tests_run}")
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} tests failed")
            return False

    def get_failed_tests(self):
        """Get list of failed tests"""
        return [test for test in self.test_results if not test['success']]

def main():
    tester = TurbineFaultDetectAPITester()
    success = tester.run_all_tests()
    
    # Save detailed results
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_tests': tester.tests_run,
                'passed_tests': tester.tests_passed,
                'failed_tests': tester.tests_run - tester.tests_passed,
                'success_rate': (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': tester.test_results,
            'failed_tests': tester.get_failed_tests()
        }, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())