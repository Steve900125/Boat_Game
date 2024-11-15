import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from boat import Boat, BoatSize, BoatMoveDirection
from config import STRATEGY_MODE, ATTACKER_ID, DEFENDER_ID

class Game:
    def __init__(self, map_size: int):
        self.map_size = map_size
        self.map_grid = np.full((map_size, map_size), None, dtype=object)
        self.attacker_boats = []
        self.defender_boats = []

    def create_map(self) -> np.ndarray:
        """Initialize the game map."""
        self.map_grid = np.full((self.map_size, self.map_size), None, dtype=object)
        return self.map_grid
    
    def print_map_grid(self):
        """Print the current map grid configuration with boat names and sizes."""
        for row in range(self.map_size):
            row_display = []
            for col in range(self.map_size):
                cell = self.map_grid[row, col]
                if cell is None:
                    row_display.append("[  ]")  # Empty cell
                else:
                    # Display boat name and size, formatted to fit within cell
                    boat_name = cell.name[:1]  # Display first 2 characters of the name
                    boat_size = cell.boat_size.value[0]  # Display first character of the size
                    row_display.append(f"[{boat_name}{boat_size}]")
            print(" ".join(row_display))

    def place_boat(self, boat: Boat, x: int, y: int) -> bool:
        """Place a boat on the map at the specified coordinates."""
        if 0 <= x < self.map_size and 0 <= y < self.map_size and self.map_grid[x, y] is None:
            self.map_grid[x, y] = boat
            boat.loc = {'x': x, 'y': y}
            print(f"Placed {boat.name} at ({x}, {y})")
            return True
        print(f"Cannot place {boat.name} at ({x}, {y}): Position occupied or out of bounds.")
        return False
    
    def place_all_boats(self, boats: List[Boat], coordinates: Optional[Dict[Boat, Tuple[int, int]]] = None):
        """Place all boats either manually or based on predefined coordinates."""
        if STRATEGY_MODE and coordinates:
            # Place boats using predefined coordinates from a dictionary
            for boat in boats:
                if boat in coordinates:
                    x, y = coordinates[boat]
                    self.place_boat(boat, x, y)
                else:
                    print(f"No coordinates provided for {boat.name}. Skipping placement.")
        else:
            # Manually input coordinates for each boat
            for boat in boats:
                while True:
                    try:
                        coords = input(f"Enter coordinates for {boat.name} (format: x y): ")
                        x, y = map(int, coords.split())
                        if self.place_boat(boat, x, y):
                            break
                    except ValueError:
                        print("Invalid input. Please enter coordinates in 'x y' format.")

    def create_attacker(self, user_id: str, names: List[str]) -> List[Boat]:
        """Create boats for the attacker and add them to the game."""
        self.attacker_boats = [
            Boat(BoatSize.LARGE, user_id, name=names[i]) for i in range(4)
        ] + [
            Boat(BoatSize.MEDIUM, user_id, name=names[i]) for i in range(4, 6)
        ] + [
            Boat(BoatSize.SMALL, user_id, name=names[i]) for i in range(6, 8)
        ]
        return self.attacker_boats
    
    def create_defender(self, user_id: str, names: List[str]) -> List[Boat]:
        """Create boats for the defender and add them to the game."""
        self.defender_boats = [
            Boat(BoatSize.LARGE, user_id, name=names[i]) for i in range(3)
        ] + [
            Boat(BoatSize.MEDIUM, user_id, name=names[i]) for i in range(3, 6)
        ] + [
            Boat(BoatSize.SMALL, user_id, name=names[i]) for i in range(6, 9)
        ]
        return self.defender_boats
    
    def execute_all_move(self, move_steps: Dict[Boat, List[BoatMoveDirection]]):
        """Executes movement actions for all boats based on provided move steps.

        Args:
            move_steps: A dictionary where each key is a `Boat` object and the value is a list of 
                        `BoatMoveDirection` representing the steps the boat should take.

        Raises:
            ValueError: If the provided movement steps are invalid or exceed allowed movement range.
        """
        for boat, moves in move_steps.items():
            if moves:
                print(f"Executing moves for {boat.name}: {moves}")
                try:
                    movement_success = boat.move(moves, self.map_grid)
                    if movement_success:
                        print(f"{boat.name} moved successfully.")
                    else:
                        print(f"{boat.name} could not complete its movement.")
                except ValueError as e:
                    print(f"Error during movement of {boat.name}: {e}")
            else:
                print(f"{boat.name} has no moves to execute.")

    def execute_all_attack(self, attack_steps: Optional[Dict[Boat, Optional[Dict[str, Any]]]] = None):
        """Executes attack actions for all boats based on provided attack steps.

        Args:
            attack_steps: A dictionary where each key is a `Boat` object and the value is an optional dictionary 
                        containing attack details:
                        - `weapon_type`: The type of weapon to use for the attack.
                        - `loc`: A dictionary specifying the target location with 'x' and 'y' coordinates.
        
        Raises:
            ValueError: If the boat's owner ID is unrecognized.
        """
        if attack_steps is None:
            return 
        
        for boat, step in attack_steps.items():
            if step is not None:
                weapon_type = step['weapon_type']
                target_loc = step['loc']

                # Determine enemy boats based on ownership
                if boat.owner_id == ATTACKER_ID:
                    enemy_boats = self.defender_boats
                elif boat.owner_id == DEFENDER_ID:
                    enemy_boats = self.attacker_boats
                else:
                    raise ValueError(f"Unrecognized owner ID: {boat.owner_id}")

                # Execute the attack
                attack_success = boat.attack(weapon_type, target_loc, self.map_grid, enemy_boats)
                
                # Feedback based on the attack result
                if attack_success:
                    print(f"{boat.name} successfully attacked {target_loc} with {weapon_type.name}.")
                else:
                    print(f"{boat.name} failed to attack {target_loc}.")

    def round(self, move_steps: Dict[Boat, List[BoatMoveDirection]], attack_steps: Dict[Boat, Optional[Dict[str, Any]]]):
        self.execute_all_move(move_steps)
        self.execute_all_attack(attack_steps)
