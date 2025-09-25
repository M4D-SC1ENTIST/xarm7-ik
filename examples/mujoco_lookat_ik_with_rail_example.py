import numpy as np
from xarm7_ik.solver import LookAtInverseKinematicsSolver
import mujoco
import mujoco.viewer
import os
import time

# Path to MJCF model with linear rail
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xarm7_env/mjcf/xarm7_camera_rail.xml")

class LookAtVisualizationDemo:
    def __init__(self):
        # Create Look-At IK solver instance (with rail)
        self.lookat_solver = LookAtInverseKinematicsSolver(
            use_linear_motor=True,
            linear_motor_x_offset=0.0,
            rotation_repr="quaternion",
            lookat_offset=np.array([0.0, 0.0, -0.15])  # Look direction relative to gripper
        )
        
        # Load Mujoco model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        
        # Initial joint configuration (8 DOF: rail + 7 joints)
        self.initial_configuration = np.array([0.0,4.11079314, 0.87910421, -0.01052001, 1.59630604, 1.86509204, -0.90185576, -0.92323043,])
        
        # Demo scenarios
        self.current_scenario = 0
        self.scenarios = self.create_scenarios()
        
        # Add visualization spheres to the model
        self.add_visualization_spheres()
        
        print("=== Look-At IK MuJoCo Visualization Demo ===")
        print("This demo shows look-at directions with fixed gripper position")
        print("The robot maintains the same target position while changing look directions:")
        print("  - Look Right/Left (horizontal scanning)")
        print("  - Look Up/Down (vertical scanning)")  
        print("  - Look Diagonal (combined directions)")
        print("\nVisualization:")
        print("  - GREEN sphere: Target gripper position")
        print("  - RED sphere: Look-at target position")
        print("  - BLUE line: Look direction from gripper")
        print("\nControls:")
        print("  SPACE: Next scenario")
        print("  R: Reset to initial position")
        print("  ESC: Exit")
        print(f"Total scenarios: {len(self.scenarios)}")

    def create_scenarios(self):
        """Create different demo scenarios with fixed position and varying look directions"""
        scenarios = []
        
        # Fixed target position (default comfortable position for the robot)
        fixed_target_pos = np.array([0.009, 0.129, 0.562])
        
        # Scenario 1: Look Forward
        scenarios.append({
            'name': 'Look Forward',
            'target_pos': fixed_target_pos,
            'lookat_pos': np.array([0.7, 0, 0.5]),  # Same height, to the right
            'description': 'Look Forward'
        })
        
        # Scenario 2: Look Backward  
        scenarios.append({
            'name': 'Look Backward',
            'target_pos': fixed_target_pos,
            'lookat_pos': np.array([-0.7, 0, 0.562]),   # Same height, to the left
            'description': 'Look Backward'
        })
        
        # Scenario 3: Look Up
        scenarios.append({
            'name': 'Look Up',
            'target_pos': fixed_target_pos,
            'lookat_pos': np.array([0.7, 0.0, 0.8]),   # Same X,Y, higher Z
            'description': 'Robot looks upward while maintaining position'
        })
        
        # Scenario 4: Look Down
        scenarios.append({
            'name': 'Look Down',
            'target_pos': fixed_target_pos,
            'lookat_pos': np.array([0.7, 0.0, 0.1]),   # Same X,Y, lower Z
            'description': 'Robot looks downward while maintaining position'
        })
        
        # Scenario 5: Look Forward-Right-Up (diagonal)
        scenarios.append({
            'name': 'Look Diagonal Up-Right',
            'target_pos': fixed_target_pos,
            'lookat_pos': np.array([0.9, -0.3, 0.6]),  # Forward, right, and up
            'description': 'Robot looks diagonally up and to the right'
        })
        
        # Scenario 6: Look Backward-Left-Down (opposite diagonal)
        scenarios.append({
            'name': 'Look Diagonal Down-Left',
            'target_pos': fixed_target_pos,
            'lookat_pos': np.array([0.5, 0.3, 0.2]),   # Backward, left, and down
            'description': 'Robot looks diagonally down and to the left'
        })
        
        return scenarios

    def add_visualization_spheres(self):
        """Add visualization spheres to the MuJoCo model"""
        # We'll use the viewer's built-in visualization features
        # Initialize sphere positions (will be updated in scenarios)
        self.target_sphere_pos = np.array([0.0, 0.0, 0.0])
        self.lookat_sphere_pos = np.array([0.0, 0.0, 0.0])
        print("Visualization spheres initialized")

    def update_marker_positions(self):
        """Update the positions of the marker spheres in the MuJoCo model"""
        try:
            # Find the marker bodies and update their positions
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
            lookat_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lookat_marker")
            
            if target_body_id >= 0:
                # Update target marker position
                self.model.body_pos[target_body_id] = self.target_sphere_pos
                
            if lookat_body_id >= 0:
                # Update lookat marker position  
                self.model.body_pos[lookat_body_id] = self.lookat_sphere_pos
                
            # Print positions for console feedback
            # Use the correct body name - based on the error, it should be one of the link bodies
            try:
                gripper_pos = self.data.xpos[self.model.body("camera_body").id]
            except:
                try:
                    gripper_pos = self.data.xpos[self.model.body("link7").id]  # Last link as end-effector
                except:
                    gripper_pos = np.array([0,0,0])
                    
            print(f"🟢 Target (GREEN): [{self.target_sphere_pos[0]:.3f}, {self.target_sphere_pos[1]:.3f}, {self.target_sphere_pos[2]:.3f}]")
            print(f"🔴 Look-at (RED):  [{self.lookat_sphere_pos[0]:.3f}, {self.lookat_sphere_pos[1]:.3f}, {self.lookat_sphere_pos[2]:.3f}]")
            print(f"🤖 Gripper:       [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
            
        except Exception as e:
            print(f"Could not update marker positions: {e}")
            # Fallback to console output only
            try:
                gripper_pos = self.data.xpos[self.model.body("camera_body").id]
            except:
                try:
                    gripper_pos = self.data.xpos[self.model.body("link7").id]
                except:
                    gripper_pos = np.array([0,0,0])
                    
            print(f"🟢 Target (GREEN): [{self.target_sphere_pos[0]:.3f}, {self.target_sphere_pos[1]:.3f}, {self.target_sphere_pos[2]:.3f}]")
            print(f"🔴 Look-at (RED):  [{self.lookat_sphere_pos[0]:.3f}, {self.lookat_sphere_pos[1]:.3f}, {self.lookat_sphere_pos[2]:.3f}]")
            print(f"🤖 Gripper:       [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")

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
        print(f"Rail position: {ik_result[0]:.4f} m")
        print(f"Joint angles: {ik_result[1:]}")
        
        # Verify the result
        achieved_pos, achieved_quat = self.lookat_solver.forward_kinematics(ik_result)
        pos_error = np.linalg.norm(achieved_pos - target_pos)
        
        # Calculate look-at error
        from xarm7_ik.utils import calculate_look_at_error
        lookat_error = calculate_look_at_error(achieved_pos, lookat_pos, achieved_quat, self.lookat_solver.lookat_offset)
        
        print(f"Position error: {pos_error:.6f} m")
        print(f"Look-at error: {lookat_error:.6f}")
        
        # Update MuJoCo simulation
        # Adjust rail position for MuJoCo (subtract initial offset)
        adjusted_ik_result = ik_result.copy()
        adjusted_ik_result[0] -= 0.37
        
        # Set joint positions
        for i in range(8):
            self.data.qpos[i] = adjusted_ik_result[i]
        
        # Step simulation to update geometry
        mujoco.mj_forward(self.model, self.data)
        
        # Update visualization sphere positions
        self.target_sphere_pos = target_pos.copy()
        self.lookat_sphere_pos = lookat_pos.copy()
        
        return scenario_idx

    def reset_to_initial(self):
        """Reset robot to initial configuration"""
        print("\nResetting to initial position...")
        
        # Set initial configuration
        for i in range(8):
            self.data.qpos[i] = 0.0 if i > 0 else 0.0  # Rail at 0, joints at 0
        
        # Step simulation
        mujoco.mj_forward(self.model, self.data)

    def run_demo(self):
        """Run the interactive MuJoCo demo"""
        # Start with first scenario
        self.solve_and_visualize_scenario(self.current_scenario)
        
        # Launch Mujoco viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print(f"\nStarting with scenario: {self.scenarios[self.current_scenario]['name']}")
            print("Press SPACE for next scenario, R to reset, ESC to exit")
            
            last_key_time = 0
            key_debounce = 0.3  # Prevent key repeat
            
            while viewer.is_running():
                current_time = time.time()
                
                # Check for key presses (simplified - in real MuJoCo you'd use proper key handling)
                # This is a simplified approach - actual key handling would depend on viewer implementation
                
                # Auto-cycle through scenarios for demo (every 8 seconds)
                if current_time - last_key_time > 8.0:
                    self.current_scenario = (self.current_scenario + 1) % len(self.scenarios)
                    self.solve_and_visualize_scenario(self.current_scenario)
                    last_key_time = current_time
                
                viewer.sync()

    def run_all_scenarios_sequence(self):
        """Run all scenarios in sequence for demonstration"""
        print("\n=== Running All Scenarios in Sequence ===")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for i, scenario in enumerate(self.scenarios):
                if not viewer.is_running():
                    break
                    
                print(f"\nDisplaying scenario {i+1}/{len(self.scenarios)}")
                self.solve_and_visualize_scenario(i)
                
                # Update marker positions and display each scenario for 5 seconds
                self.update_marker_positions()
                start_time = time.time()
                while time.time() - start_time < 5.0 and viewer.is_running():
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
        demo = LookAtVisualizationDemo()
        
        # Choose demo mode
        print("\nDemo modes:")
        print("1. Interactive demo (manual control)")
        print("2. Automatic sequence (all scenarios)")
        
        # For this example, we'll run the automatic sequence
        # You can modify this to add interactive controls
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
