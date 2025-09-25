import numpy as np
from xarm7_ik.kinematics import LookAtInverseKinematicsSolver
import mujoco
import mujoco.viewer
import os
import time

# Path to MJCF model (without linear rail)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xarm7_env/mjcf/xarm7.xml")

class SimpleLookAtVisualizationDemo:
    def __init__(self):
        # Create Look-At IK solver instance (without rail)
        self.lookat_solver = LookAtInverseKinematicsSolver(
            use_linear_motor=False,
            rotation_repr="quaternion",
            base_rotation_offset=-np.pi/2,  # Match the MuJoCo model orientation
            lookat_offset=np.array([0.0, 0.0, -0.15])  # Look direction relative to gripper
        )
        
        # Load Mujoco model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        
        # Initial joint configuration (7 DOF)
        self.initial_configuration = np.zeros(7)
        
        # Demo scenarios (within reach of fixed-base robot)
        self.scenarios = self.create_scenarios()
        
        print("=== Look-At IK MuJoCo Visualization Demo (Fixed Base) ===")
        print("This demo shows the look-at IK solver without linear rail")
        print(f"Total scenarios: {len(self.scenarios)}")

    def create_scenarios(self):
        """Create different demo scenarios for fixed-base robot"""
        scenarios = []
        
        # Scenario 1: Basic look-at demonstration
        scenarios.append({
            'name': 'Basic Look-At',
            'target_pos': np.array([0, 0.2, 0.3]),
            'lookat_pos': np.array([0.0, 0.0, 0.1]),
            'description': 'Basic look-at constraint demonstration'
        })
        
        # Scenario 2: Look at side object
        scenarios.append({
            'name': 'Side Object Tracking',
            'target_pos': np.array([0.4, 0.3, 0.4]),
            'lookat_pos': np.array([-0.2, 0.4, 0.2]),
            'description': 'Looking at object to the side'
        })
        
        # Scenario 3: Overhead inspection
        scenarios.append({
            'name': 'Overhead Inspection',
            'target_pos': np.array([0.3, 0.0, 0.5]),
            'lookat_pos': np.array([0.3, 0.0, 0.0]),
            'description': 'Overhead positioning looking straight down'
        })
        
        # Scenario 4: Angular positioning
        scenarios.append({
            'name': 'Angular Positioning',
            'target_pos': np.array([0.6, -0.2, 0.3]),
            'lookat_pos': np.array([0.2, 0.3, 0.15]),
            'description': 'Angular positioning with look-at constraint'
        })
        
        # Scenario 5: Close inspection
        scenarios.append({
            'name': 'Close Inspection',
            'target_pos': np.array([0.4, 0.1, 0.25]),
            'lookat_pos': np.array([0.35, 0.05, 0.2]),
            'description': 'Close-up inspection task'
        })
        
        return scenarios

    def solve_and_visualize_scenario(self, scenario_idx):
        """Solve IK for a scenario and update visualization"""
        if scenario_idx >= len(self.scenarios):
            scenario_idx = 0
            
        scenario = self.scenarios[scenario_idx]
        target_pos = scenario['target_pos']
        lookat_pos = scenario['lookat_pos']
        
        print(f"\n--- Scenario {scenario_idx + 1}: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Target position: {target_pos}")
        print(f"Look-at position: {lookat_pos}")
        
        # Solve Look-At IK
        start_time = time.time()
        ik_result = self.lookat_solver.inverse_kinematics(
            self.initial_configuration,
            target_pos,
            lookat_pos
        )
        solve_time = time.time() - start_time
        
        print(f"IK solve time: {solve_time*1000:.2f} ms")
        print(f"Joint angles: {ik_result}")
        
        # Verify the result
        achieved_pos, achieved_quat = self.lookat_solver.forward_kinematics(ik_result)
        pos_error = np.linalg.norm(achieved_pos - target_pos)
        
        # Calculate look-at error
        from xarm7_ik.utils import calculate_look_at_error
        lookat_error = calculate_look_at_error(achieved_pos, lookat_pos, achieved_quat, self.lookat_solver.lookat_offset)
        
        print(f"Position error: {pos_error:.6f} m")
        print(f"Look-at error: {lookat_error:.6f}")
        
        # Update MuJoCo simulation
        # Set joint positions
        for i in range(7):
            self.data.qpos[i] = ik_result[i]
        
        # Step simulation to update geometry
        mujoco.mj_forward(self.model, self.data)
        
        return scenario_idx

    def reset_to_initial(self):
        """Reset robot to initial configuration"""
        print("\nResetting to initial position...")
        
        # Set initial configuration
        for i in range(7):
            self.data.qpos[i] = 0.0
        
        # Step simulation
        mujoco.mj_forward(self.model, self.data)

    def run_all_scenarios_sequence(self):
        """Run all scenarios in sequence for demonstration"""
        print("\n=== Running All Scenarios in Sequence ===")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for i, scenario in enumerate(self.scenarios):
                if not viewer.is_running():
                    break
                    
                print(f"\nDisplaying scenario {i+1}/{len(self.scenarios)}")
                self.solve_and_visualize_scenario(i)
                
                # Display each scenario for 4 seconds
                start_time = time.time()
                while time.time() - start_time < 4.0 and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)  # Small delay to prevent busy waiting
            
            # Keep viewer open at the end
            if viewer.is_running():
                print("\nDemo complete. Viewer will remain open. Press ESC to exit.")
                while viewer.is_running():
                    viewer.sync()

def main():
    """Main function to run the demo"""
    try:
        demo = SimpleLookAtVisualizationDemo()
        demo.run_all_scenarios_sequence()
        
    except FileNotFoundError:
        print(f"Error: Could not find MuJoCo model file at {MODEL_PATH}")
        print("Make sure the xarm7_env directory exists with the MJCF files.")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
