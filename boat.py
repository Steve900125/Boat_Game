from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Self
import numpy as np

class BoatSize(Enum):
    """Represents the size of the boat."""
    SMALL: str = "small"
    MEDIUM: str = "medium"
    LARGE: str = "large"

class BoatMoveDirection(Enum):
    """Represents possible movement directions for boats."""
    UP: str = 'up'
    DOWN: str = 'down'
    LEFT: str = 'left'
    RIGHT: str = 'right'

class WeaponType(Enum):
    """Represents the types of weapons available for boats."""
    M_SONIC = "sonic"
    M_HYPERSONIC = "hypersonic"

@dataclass
class BoatConfig:
    """Configuration for a boat's properties.

    Attributes:
        sonic_missile_count: The number of sonic missiles the boat has.
        hypersonic_missile_count: The number of hypersonic missiles the boat has.
        hp: The health points of the boat.
        mvdist: The maximum movement distance for the boat.
    """
    sonic_missile_count: int
    hypersonic_missile_count: int
    hp: int
    mvdist: int

class Boat:
    """Represents a boat with specific configurations and behaviors.

    Attributes:
        boat_setting: A dictionary that provides configurations for each boat size.
    """

    boat_setting = {
        BoatSize.SMALL: BoatConfig(sonic_missile_count=0, hypersonic_missile_count=3, hp=1, mvdist=2),
        BoatSize.MEDIUM: BoatConfig(sonic_missile_count=4, hypersonic_missile_count=2, hp=2, mvdist=3),
        BoatSize.LARGE: BoatConfig(sonic_missile_count=5, hypersonic_missile_count=4, hp=3, mvdist=4)
    }

    def __init__(self, boat_size: BoatSize, owner_id: str, loc: Optional[Dict[str, int]] = None, name: Optional[str] = None):
        """Initializes a boat with given size, owner, and optional location and name.

        Args:
            boat_size: Size of the boat.
            owner_id: The ID of the boat's owner.
            loc: Initial location of the boat on the map.
            name: Optional name for the boat.

        Raises:
            ValueError: If the provided boat size is invalid.
        """
        if not isinstance(boat_size, BoatSize):
            raise ValueError('Boat size must be an instance of BoatSize(enum)')
        
        self.boat_size = boat_size
        self.owner_id = owner_id
        config = self.boat_setting.get(boat_size)

        if not config:
            raise ValueError(f"Invalid boat size: {boat_size}")

        self.sonic_missile_count = config.sonic_missile_count
        self.hypersonic_missile_count = config.hypersonic_missile_count
        self.hp = config.hp
        self.mvdist = config.mvdist

        self.loc = loc if loc else {'x': -1, 'y': -1}
        self.name = name or f"{self.owner_id}'s Mighty Ship"

    def __str__(self) -> str:
        """Returns a string representation of the boat."""
        return (f"Boat size: {self.boat_size.value}, Owner ID: {self.owner_id}, "
                f"Sonic Missiles: {self.sonic_missile_count}, Hypersonic Missiles: {self.hypersonic_missile_count}, "
                f"HP: {self.hp}, Move Distance: {self.mvdist}, Location: {self.loc}")
    
    # Move on map
    def move(self, move_steps: List[BoatMoveDirection], map_grid: np.ndarray) -> bool:
        """Moves the boat according to a list of directions within map boundaries.

        Args:
            move_steps: List of directions for the boat to move in sequence.
            map_grid: 2D numpy array representing the game map.

        Returns:
            True if the movement is successful; False otherwise.
        """

        # Check movement is valid
        if len(move_steps) > self.mvdist:
            print(f"Cannot move. The number of steps ({len(move_steps)}) exceeds the maximum allowed ({self.mvdist}).")
            return False
        elif move_steps is None:
            print(f"The move_steps is None")
            return True
        
        # Start moving
        for step in move_steps:
            print(move_steps)
            new_x, new_y = self.loc['x'], self.loc['y']
            if step == BoatMoveDirection.UP:
                new_x -= 1
            elif step == BoatMoveDirection.DOWN:
                new_x += 1
            elif step == BoatMoveDirection.LEFT:
                new_y -= 1
            elif step == BoatMoveDirection.RIGHT:
                new_y += 1

            # Check it still in the map
            if not (0 <= new_x < map_grid.shape[0] and 0 <= new_y < map_grid.shape[1]):
                print(f"Cannot move {step.value}. Out of map bounds.")
                continue
            
            # Check the path is occupied
            if map_grid[new_x, new_y] is not None:
                print(f"Cannot move {step.value}. Position ({new_x}, {new_y}) is occupied.")
                continue
        
            # Update the boat to new location
            map_grid[self.loc['x'], self.loc['y']] = None
            self.loc['x'], self.loc['y'] = new_x, new_y
            map_grid[new_x, new_y] = self
            print(f"Boat {self.name} moved {step.value} to ({new_x}, {new_y})")

        return True
    
    def get_attack_range(self, weapon_type: WeaponType) -> List[Tuple[int, int]]:
        """Gets the attack range based on weapon type.

        Args:
            weapon_type: The type of weapon to use.

        Returns:
            A list of tuples representing attack positions.
        """
        if weapon_type == WeaponType.M_SONIC:
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif weapon_type == WeaponType.M_HYPERSONIC:
            return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            return []
        
    def get_move_range(self) -> List[Tuple[int, int]]:
        """Calculates all possible moves within the boat's movement range.

        Returns:
            A list of tuples indicating all valid movement positions.
        """
        move_range = set()
        for dx in range(-self.mvdist, self.mvdist + 1):
            for dy in range(-self.mvdist, self.mvdist + 1):
                if abs(dx) + abs(dy) <= self.mvdist:
                    move_range.add((dx, dy))
        return list(move_range)

    def is_weapon_usage_valid(self, weapon_type: WeaponType) -> bool:
        """Checks if the specified weapon type can be used.

        Args:
            weapon_type: The type of weapon to check.

        Returns:
            True if the weapon can be used; False otherwise.
        """
        if weapon_type == WeaponType.M_SONIC and self.sonic_missile_count <= 0:
            return False
        if weapon_type == WeaponType.M_HYPERSONIC and self.hypersonic_missile_count <= 0:
            return False
        
        return True
    
    def is_attack_valid(self, weapon_type: WeaponType, target_loc: Dict[str, int], map_grid: np.ndarray) -> bool:
        """Checks if an attack is valid without harming teammates.

        Args:
            weapon_type (WeaponType): The type of weapon used for the attack.
            target_loc (Dict[str, int]): The target location of the attack, 
                specified as a dictionary with 'x' and 'y' coordinates.
            map_grid (np.ndarray): The 2D numpy array representing the game map.

        Returns:
            bool: True if the attack is valid (does not harm teammates and stays 
                within boundaries); False otherwise.
        """
        origin_x, origin_y = target_loc['x'], target_loc['y']
        attack_positions = self.get_attack_range(weapon_type)

        for dx, dy in attack_positions:
            target_x, target_y = origin_x + dx, origin_y + dy
            if 0 <= target_x < map_grid.shape[0] and 0 <= target_y < map_grid.shape[1]:
                target = map_grid[target_x, target_y]
                if target is not None and target.owner_id == self.owner_id:
                    return False
        return True
    
    def consume_weapon(self, weapon_type: WeaponType):
        """Decreases the count of the specified weapon after an attack.

        Args:
            weapon_type: The type of weapon used.
        """
        if weapon_type == WeaponType.M_SONIC:
            self.sonic_missile_count -= 1
        elif weapon_type == WeaponType.M_HYPERSONIC:
            self.hypersonic_missile_count -= 1

    def damage_calc(self, weapon_type: WeaponType, target_loc: Dict[str, int], map_grid: np.ndarray, enemy_boats: 'List[Boat]'):
        """Calculates and applies damage to enemy boats within the attack range.

        This function determines which boats are within the attack range based on 
        the weapon type and applies damage to them. If a boat's HP reaches zero, 
        it is removed from the map and the list of enemy boats.

        Args:
            weapon_type (WeaponType): The type of weapon used for the attack.
            target_loc (Dict[str, int]): The central target location of the attack, 
                specified as a dictionary with 'x' and 'y' coordinates.
            map_grid (np.ndarray): The 2D numpy array representing the game map.
            enemy_boats (List[Boat]): The list of enemy boats on the map.
        """
        attack_positions = self.get_attack_range(weapon_type)
        origin_x, origin_y = target_loc['x'], target_loc['y']

        for dx, dy in attack_positions:
            target_x, target_y = origin_x + dx, origin_y + dy
            if 0 <= target_x < map_grid.shape[0] and 0 <= target_y < map_grid.shape[1]:
                target = map_grid[target_x, target_y]
                if target and target in enemy_boats:
                    target.hp -= 1
                    print(f"Damage to {target.name} successful. Remaining HP: {target.hp}")

                    if target.hp <= 0:
                        map_grid[target_x, target_y] = None
                        enemy_boats.remove(target)
                        print(f"Enemy {target.name} is down!")


    def attack(self, weapon_type: WeaponType, target_loc: Dict[str, int], map_grid: np.ndarray, enemy_boats: 'List[Boat]') -> bool:
        """Executes an attack action on the target location.

        The attack checks weapon validity, ensures friendly boats are not harmed, 
        consumes the weapon, and applies damage to enemy boats.

        Args:
            weapon_type (WeaponType): The type of weapon used.
            target_loc (Dict[str, int]): The location to attack, specified as a 
                dictionary with 'x' and 'y' coordinates.
            map_grid (np.ndarray): The 2D numpy array representing the game map.
            enemy_boats (List[Boat]): The list of enemy boats on the map.

        Returns:
            bool: True if the attack is executed successfully; False otherwise.
        """
        if not self.is_weapon_usage_valid(weapon_type):
            print("Doesn't have weapon can be used")
            return False

        if not self.is_attack_valid(weapon_type, target_loc, map_grid):
            print("Attack aborted to avoid friendly fire.")
            return False

        self.consume_weapon(weapon_type)
        self.damage_calc(weapon_type, target_loc, map_grid, enemy_boats)
        return True
