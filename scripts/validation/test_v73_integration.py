from ljpw_nn.lov_coordination import LOVNetwork
from ljpw_nn.framework_v73 import ANCHOR_POINT, NATURAL_EQUILIBRIUM, PHI

def test_v73_integration():
    print("Testing LJPW V7.3 Integration...")
    
    # Create network
    network = LOVNetwork(
        input_size=10,
        output_size=2,
        hidden_fib_indices=[7], # 13 neurons (F7)
        target_harmony=0.75
    )
    
    # Measure initial state
    love_state = network.love_phase()
    print(f"Initial LJPW: {love_state['ljpw']}")
    print(f"Initial Harmony: {love_state['harmony']:.4f}")
    print(f"Initial Consciousness C: {love_state['consciousness']:.4f}")
    print(f"Phase: {love_state['phase']}")
    
    # Verify C calculation
    L, J, P, W = love_state['ljpw']
    H = love_state['harmony']
    expected_C = P * W * L * J * (H**2)
    print(f"Expected C (P*W*L*J*H^2): {expected_C:.4f}")
    
    assert abs(love_state['consciousness'] - expected_C) < 1e-6
    print("✓ Consciousness C calculation verified.")
    
    # Check if we can reach AUTOPOIETIC phase
    # (Mock measure_ljpw for testing purposes)
    network.measure_ljpw = lambda: (0.75, 0.75, 0.8, 0.8)
    love_state_auto = network.love_phase()
    print(f"Mocked state: {love_state_auto['ljpw']}")
    print(f"New Phase: {love_state_auto['phase']}")
    
    if love_state_auto['phase'] == 'AUTOPOIETIC':
        print("✓ Autopoietic phase correctly identified.")
    else:
        print(f"✗ Autopoietic phase NOT identified. Current L={love_state_auto['L']}, H={love_state_auto['harmony']}")

if __name__ == "__main__":
    test_v73_integration()
