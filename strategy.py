from boat import Boat, BoatSize, BoatMoveDirection,  WeaponType
from typing import List, Tuple, Dict, Optional, Any
from config import MAP_SIZE, ATTACKER_ID, DEFENDER_ID
from game import Game
from math import sqrt
import numpy as np 
import random

class Strategy:
    @staticmethod
    def place_widest(boats: List[Boat], is_attacker: bool) -> Dict[Boat, Tuple[int, int]]:
        """
        Place boats on the map in a grid pattern, centered around a specified quadrant midpoint.
        - Attacker boats are centered in the top-left quadrant.
        - Defender boats are centered in the bottom-right quadrant.
        - Distributes boats based on size, with larger boats closer to the center.
        - Returns a dictionary with Boat as the key and its coordinates as the value.
        """
        coordinates = {}
        
        # Define center positions for attacker and defender quadrants
        center_x, center_y = ((MAP_SIZE) // 4 ,(MAP_SIZE-1) // 2) if is_attacker else (3* (MAP_SIZE-1)// 4 , (MAP_SIZE-1) // 2)
        offset_high = (MAP_SIZE-2)// 5 # Offset for expanding placement positions around the center
        offset_wide =  (MAP_SIZE-3)// 3

        # Define a grid pattern around the selected center
        positions = [
            (center_x, center_y),                       # Center
            (center_x, center_y + offset_wide),              # Right
            (center_x, center_y - offset_wide),              # Left
            (center_x +  offset_high , center_y),              # Bottom
            (center_x -  offset_high , center_y),              # Top
            (center_x -  offset_high , center_y + offset_wide),     # Top-right
            (center_x -  offset_high , center_y - offset_wide),      # Top-left
            (center_x +  offset_high , center_y + offset_wide),     # Bottom-right
            (center_x +  offset_high , center_y - offset_wide)      # Bottom-left
        ]

        # Separate boats by size to place larger ones closer to the center
        large_boats = [boat for boat in boats if boat.boat_size == BoatSize.LARGE]
        medium_boats = [boat for boat in boats if boat.boat_size == BoatSize.MEDIUM]
        small_boats = [boat for boat in boats if boat.boat_size == BoatSize.SMALL]
        sorted_boats = large_boats + medium_boats + small_boats  # Order by size
        
        # Assign positions to boats in a grid pattern
        for boat, pos in zip(sorted_boats, positions):
            coordinates[boat] = pos

        return coordinates
    
    @staticmethod
    def move_randomly(boat: Boat, map_grid: np.ndarray, move_stack: Optional[List[Tuple[int, int]]] = None, max_movement: bool = True) -> Optional[List[BoatMoveDirection]]:
        """
        Generate a list of random movement directions for a boat based on its max movement distance (mvdist).
        - Avoids revisiting positions recorded in `move_stack`.
        - `max_movement`: if True, ensures the boat moves the maximum possible steps.
        - Returns a list of valid moves or None if no valid moves are possible.
        """
        moves = []
        directions = [BoatMoveDirection.UP, BoatMoveDirection.DOWN, 
                      BoatMoveDirection.LEFT, BoatMoveDirection.RIGHT, None]
        execute_counter = 0
        current_x, current_y = boat.loc['x'], boat.loc['y']
        previous_move = None  # Track the last move direction

        # Initialize `move_stack` if it is None
        move_stack = move_stack if move_stack is not None else []
        visited_positions = set(move_stack) | {(current_x, current_y)}  # Include `move_stack` to prevent revisiting
        max_attempts = 10 * boat.mvdist  # Set a reasonable max attempt limit to avoid infinite loops

        # Define opposite directions to avoid backtracking
        opposite_directions = {
            BoatMoveDirection.UP: BoatMoveDirection.DOWN,
            BoatMoveDirection.DOWN: BoatMoveDirection.UP,
            BoatMoveDirection.LEFT: BoatMoveDirection.RIGHT,
            BoatMoveDirection.RIGHT: BoatMoveDirection.LEFT
        }

        while execute_counter < boat.mvdist and max_attempts > 0:
            # Filter out the opposite of the previous move to avoid backtracking
            valid_directions = [d for d in directions if d != opposite_directions.get(previous_move)]
            random.shuffle(valid_directions)  # Randomize direction order each time

            move_found = False  # Track if any valid move is found in this iteration

            for move in valid_directions:
                if move is None:
                    # Skip this step if "no movement" is chosen and max_movement is True
                    if max_movement:
                        continue
                    else:
                        move_found = True
                        break

                new_x, new_y = current_x, current_y

                # Calculate new position based on the current direction
                if move == BoatMoveDirection.UP:
                    new_x -= 1
                elif move == BoatMoveDirection.DOWN:
                    new_x += 1
                elif move == BoatMoveDirection.LEFT:
                    new_y -= 1
                elif move == BoatMoveDirection.RIGHT:
                    new_y += 1

                # Check if the new position is within bounds, unoccupied, and not revisiting the same cell
                if ((0 <= new_x < map_grid.shape[0] and 
                     0 <= new_y < map_grid.shape[1] and 
                     map_grid[new_x, new_y] is None) and
                    (new_x, new_y) not in visited_positions):
                    moves.append(move)  # Add valid move
                    previous_move = move  # Update the previous move to the current direction
                    current_x, current_y = new_x, new_y  # Update current position
                    visited_positions.add((new_x, new_y))  # Mark this position as visited
                    move_found = True
                    execute_counter += 1
                    break  # Stop checking directions for this step

            # Reduce max_attempts with each iteration to avoid infinite loop
            max_attempts -= 1

            # Exit loop if no valid moves were found in the current iteration and max_movement is False
            if not move_found and not max_movement:
                break

        # Return the moves list if it contains valid moves; otherwise, return None
        return moves if moves else None
    
    @staticmethod
    def all_move_randomly(boats: List[Boat], map_grid: np.ndarray) -> Dict[Boat, Optional[List[BoatMoveDirection]]]:
        """
        Executes random moves for each boat in `boats` on `map_grid`.
        - Returns a dictionary with each boat and its corresponding move list (or None if no moves are possible).
        
        Parameters:
            - boats: List of boats to move.
            - map_grid: The game map as a 2D numpy array.
        
        Returns:
            - Dictionary with each boat as the key and its list of moves (or None if no valid moves).
        """
        move_stack = []  # Track visited positions across all boats to avoid overlap
        all_moves = {}

        for boat in boats:
            # Generate random moves for each boat
            moves = Strategy.move_randomly(boat, map_grid, move_stack)
            all_moves[boat] = moves  # Store moves for each boat

            # If the boat moved, update move_stack with the new end position
            if moves:
                # Calculate the boat's final position after applying all moves
                end_x, end_y = boat.loc['x'], boat.loc['y']
                for move in moves:
                    if move == BoatMoveDirection.UP:
                        end_x -= 1
                    elif move == BoatMoveDirection.DOWN:
                        end_x += 1
                    elif move == BoatMoveDirection.LEFT:
                        end_y -= 1
                    elif move == BoatMoveDirection.RIGHT:
                        end_y += 1
                # Add the final position to move_stack to track it
                move_stack.append((end_x, end_y))

        return all_moves
    
    @staticmethod
    def is_boat_in_range(x: int, y: int, attack_range: List[Tuple[int, int]], 
                         boat: Boat, map_grid: np.ndarray, check_for_enemy: bool) -> bool:
        """
        Check if there's at least one boat in the attack range from (x, y), based on ownership.
        """
        for dx, dy in attack_range:
            target_x, target_y = x + dx, y + dy
            if (0 <= target_x < map_grid.shape[0] and
                0 <= target_y < map_grid.shape[1]):
                target_boat = map_grid[target_x, target_y]
                if target_boat:
                    if (check_for_enemy and target_boat.owner_id != boat.owner_id) or \
                       (not check_for_enemy and target_boat.owner_id == boat.owner_id):
                        return True
        return False

    @staticmethod
    def adjust_attack_position(boat: Boat, step: Dict[str, Any], map_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Adjust the attack target position if friendly boats are in the attack range.
        """
        target_loc = step['loc']
        weapon_type = step['weapon_type']
        attack_range = boat.get_attack_range(weapon_type)
        
        if Strategy.is_boat_in_range(target_loc['x'], target_loc['y'], attack_range, boat, map_grid, check_for_enemy=False):
            print("Friendly boats detected in attack range. Adjusting position.")
            move_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for mx, my in move_offsets:
                new_x, new_y = target_loc['x'] + mx, target_loc['y'] + my
                if (0 <= new_x < map_grid.shape[0] and
                    0 <= new_y < map_grid.shape[1] and
                    Strategy.is_boat_in_range(new_x, new_y, attack_range, boat, map_grid, check_for_enemy=True) and
                    not Strategy.is_boat_in_range(new_x, new_y, attack_range, boat, map_grid, check_for_enemy=False)):
                    print(f"Adjusted attack position to ({new_x}, {new_y}) to avoid friendly fire.")
                    return {'weapon_type': weapon_type, 'loc': {'x': new_x, 'y': new_y}}

            print("No suitable attack position found to avoid friendly fire.")
            return None

        return step

    @staticmethod
    def attack_directly(
        boat: Boat,
        enemy_boats: List[Boat],
        map_grid: np.ndarray,
        attack_records: Optional[Dict[Boat, Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Perform a direct attack on a selected enemy boat and adjust if friendly boats are in range.
        Record the attack for the current boat in the provided dictionary.
        """
        # Filter out targets already attacked based on attack_records
        if attack_records is not None:
            already_attacked_locations = {tuple(record['loc'].values()) for record in attack_records.values()}
            enemy_boats = [
                e_boat for e_boat in enemy_boats 
                if (e_boat.loc['x'], e_boat.loc['y']) not in already_attacked_locations
            ]

        if not enemy_boats:
            print("No valid targets available.")
            return None

        # Select a random target boat
        target_boat = random.choice(enemy_boats)
        target_loc = {'x': target_boat.loc['x'], 'y': target_boat.loc['y']}
        
        # Select weapon type
        weapon_type = None
        if boat.sonic_missile_count > 0 and boat.hypersonic_missile_count > 0:
            weapon_type = random.choice([WeaponType.M_SONIC, WeaponType.M_HYPERSONIC])
        elif boat.sonic_missile_count > 0:
            weapon_type = WeaponType.M_SONIC
        elif boat.hypersonic_missile_count > 0:
            weapon_type = WeaponType.M_HYPERSONIC

        if weapon_type is None:
            print("No weapons available for attack.")
            return None

        # Create attack record
        attack_record = {'weapon_type': weapon_type, 'loc': target_loc}

        # Adjust attack position to avoid friendly fire
        adjusted_attack = Strategy.adjust_attack_position(boat, attack_record, map_grid)
        if adjusted_attack:
            print(f"{boat.name} attacks at adjusted position {adjusted_attack['loc']} using {weapon_type.name}")
            
            # Record attack in attack_records dictionary
            if attack_records is not None:
                attack_records[boat] = {
                    'weapon_type': adjusted_attack['weapon_type'],
                    'loc': adjusted_attack['loc']
                }
            
            return adjusted_attack
        else:
            print("Attack aborted due to inability to adjust position without friendly fire.")
            return None


    @staticmethod
    def all_attack_directly(my_boats: List[Boat], enemy_boats: List[Boat], map_grid: np.ndarray) -> Dict[Boat, Optional[Dict[str, Any]]]:
        """
        Perform direct attacks from each boat in `my_boats` on `enemy_boats` with friendly fire checks.
        """
        attack_records = {}  # Initialize a dictionary to track attack records for all boats

        attack_results = {}
        for boat in my_boats:
            # Perform an attack and store the result
            attack_result = Strategy.attack_directly(boat, enemy_boats, map_grid, attack_records)
            attack_results[boat] = attack_result  # Record the result for the current boat

        return attack_results

    @staticmethod
    def count_weapons(boats: List[Boat]) -> int:
        """
        Calculates the total number of weapons (sonic and hypersonic missiles) for a list of boats.

        Args:
            boats: List of Boat objects.

        Returns:
            Total weapon count across all boats.
        """
        weapons_sum = 0
        for boat in boats:
            weapons_sum += boat.sonic_missile_count
            weapons_sum += boat.hypersonic_missile_count
        return weapons_sum

    @staticmethod
    def score_weapon(boats: List[Boat]) -> float:
        """
        Scores the enemy weapon usage based on the remaining weapons compared to the original total.

        Args:
            boats: List of Boat objects.

        Returns:
            A score between 0 and 10, normalized based on weapon usage.
        """
        origin_weapons_sum = 54  # Maximum weapon count (original total)
        current_weapons_sum = Strategy.count_weapons(boats)  # Current weapon count
        weapon_usage = origin_weapons_sum - current_weapons_sum  # Weapons used

        # Normalize the score to a 0-10 scale
        score = (weapon_usage / origin_weapons_sum) * 10
        return round(score, 2)  # Rounded to 2 decimal places
    
    @staticmethod
    def find_nearest_boat(main_boat: Boat, target_boats: List[Boat]) -> Optional[Boat]:
        """
        Finds the nearest boat to the main boat from the provided target boats list.

        Args:
            main_boat (Boat): The main boat.
            target_boats (List[Boat]): List of boats to evaluate.

        Returns:
            Optional[Boat]: The nearest boat, or None if no boats are found.
        """
        min_distance = float('inf')
        nearest_boat = None

        for boat in target_boats:
            if boat is not main_boat:
                distance = abs(main_boat.loc['x'] - boat.loc['x']) + abs(main_boat.loc['y'] - boat.loc['y'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_boat = boat
                    
        return nearest_boat

    @staticmethod
    def get_valid_positions(boat: Boat, map_grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Gets all valid positions a boat can move to on the map, excluding occupied positions.

        Args:
            boat (Boat): The boat to evaluate.
            map_grid (np.ndarray): The game map grid.

        Returns:
            List[Tuple[int, int]]: A list of valid positions (x, y).
        """
        valid_positions = []
        for dx, dy in boat.get_move_range():
            new_x = boat.loc['x'] + dx
            new_y = boat.loc['y'] + dy

            # Check if the position is within bounds and unoccupied
            if (
                0 <= new_x < map_grid.shape[0] and
                0 <= new_y < map_grid.shape[1] and
                map_grid[new_x, new_y] is None
            ):
                valid_positions.append((new_x, new_y))

        return valid_positions

    @staticmethod
    def score_overlap_density(main_boat: Boat, map_grid: np.ndarray, target_boats: List[Boat]) -> float:
        """
        Calculates the overlap density of paths between the main boat and surrounding boats.

        Args:
            main_boat (Boat): The main boat.
            map_grid (np.ndarray): The game map grid.
            target_boats (List[Boat]): List of target boats to evaluate.

        Returns:
            float: The overlap density score normalized to 0-10.
        """
        # Find the nearest boat
        nearest_boat = Strategy.find_nearest_boat(main_boat, target_boats)
        if not nearest_boat:
            return 0.0  # No nearby boats to overlap

        # Get valid positions for both main and nearest boats
        main_valid_positions = set(Strategy.get_valid_positions(main_boat, map_grid))
        nearest_valid_positions = set(Strategy.get_valid_positions(nearest_boat, map_grid))

        # Ensure valid positions exclude occupied spaces
        valid_main_positions = {pos for pos in main_valid_positions if map_grid[pos[0], pos[1]] is None}
        valid_nearest_positions = {pos for pos in nearest_valid_positions if map_grid[pos[0], pos[1]] is None}

        # Calculate overlap
        overlap_positions = valid_main_positions.intersection(valid_nearest_positions)
        overlap_count = len(overlap_positions)

        # Normalize the score
        max_possible_overlap = len(valid_main_positions)
        if max_possible_overlap == 0:
            return 0.0  # No valid positions, hence no score

        normalized_score = (overlap_count / max_possible_overlap) * 10
        return round(normalized_score, 2)
    
    @staticmethod
    def score_nearest_hp(main_boat: Boat, target_boats: List[Boat]) -> float:
        """
        Scores the health points (HP) of the nearest ally to the main boat.
        Lower HP of the nearest ally results in a higher score. No allies return a score of 0.

        Args:
            main_boat (Boat): The main boat.
            target_boats (List[Boat]): List of ally boats to evaluate.

        Returns:
            float: A score between 0 and 10, inversely proportional to the HP of the nearest ally.
        """
        # Find the nearest ally
        nearest_ally = Strategy.find_nearest_boat(main_boat, target_boats)
        if not nearest_ally:
            print("No nearby allies found.")
            return 0.0  # No allies to evaluate

        # Define the maximum possible HP for normalization
        max_hp = 4  # Adjust based on the game configuration

        # Inverse scoring: lower HP gives higher scores
        inverse_hp_score = max_hp - nearest_ally.hp
        normalized_score = (inverse_hp_score / max_hp) * 10
        return round(normalized_score, 2)

    
    @staticmethod
    def movement_evaluate(game: Game, user_id: str) -> Dict[Boat, Dict[str, float]]:
        """
        Evaluates the movement strategy for boats based on scoring criteria and weights.

        Movement Options:
        - Random movement
        - Move away from allies
        - Move closer to enemies

        Scoring Criteria:
        - Criterion 1: Enemy weapon usage (higher usage = higher score)
        - Criterion 2: Ally path overlap density (higher density = higher score)
        - Criterion 3: Enemy path overlap density (higher density = higher score)
        - Criterion 4: Ally HP (lower HP = higher score, no ally = 0)

        Weight Table:
            Movement/Criteria    Criterion1 Weight  Criterion2 Weight  Criterion3 Weight  Criterion4 Weight  Total Score
            Random Movement      0.4                0.3                0.2                0.1                
            Move Away from Allies 0.1               0.4                0.1                0.4                
            Move Closer to Enemies 0.1              0.2                0.4                0.3                

        Args:
            game (Game): The game instance.
            user_id (str): The user ID (attacker or defender).

        Returns:
            Dict[Boat, Dict[str, float]]: A dictionary containing the evaluated scores for each movement option for all boats.
        """
        # Determine ally and enemy boats based on the user_id
        if user_id == ATTACKER_ID:
            ally_boats = game.attacker_boats
            enemy_boats = game.defender_boats
        elif user_id == DEFENDER_ID:
            ally_boats = game.defender_boats
            enemy_boats = game.attacker_boats
        else:
            return {}

        # Weight configuration
        weights = {
            "random": [0.4, 0.3, 0.2, 0.1],
            "move_away": [0.1, 0.4, 0.1, 0.4],
            "move_closer": [0.1, 0.2, 0.5, 0.2]
        }

        evaluation_results: Dict[Boat, Dict[str, float]] = {}

        for main_boat in ally_boats:
            # Calculate scores for each criterion
            score_weapon_val = Strategy.score_weapon(enemy_boats)
            score_density_ally_val = Strategy.score_overlap_density(
                main_boat=main_boat,
                map_grid=game.map_grid,
                target_boats=ally_boats
            )
            score_density_enemy_val = Strategy.score_overlap_density(
                main_boat=main_boat,
                map_grid=game.map_grid,
                target_boats=enemy_boats
            )
            score_hp_val = Strategy.score_nearest_hp(
                main_boat=main_boat,
                target_boats=ally_boats
            )

            # Movement option evaluations
            scores = {}
            for option, weight in weights.items():
                total_score = (
                    weight[0] * score_weapon_val +
                    weight[1] * score_density_ally_val +
                    weight[2] * score_density_enemy_val +
                    weight[3] * score_hp_val
                )
                scores[option] = round(total_score, 2)

            evaluation_results[main_boat] = scores

        return evaluation_results
    
    @staticmethod
    def show_moves_evaluation_results(evaluation_results: Dict[Boat, Dict[str, float]]):
        """
        Displays the evaluation results for each boat.

        Args:
            evaluation_results (Dict[Boat, Dict[str, float]]): A dictionary where each Boat is a key,
                and the value is another dictionary containing scores for each movement option.
        """
        print("\nEvaluation Results:")
        print(f"{'Boat Name':<15} {'Random Move':<15} {'Move Away':<15} {'Move Closer':<15}")
        print("-" * 60)

        for boat, scores in evaluation_results.items():
            print(f"{boat.name:<15} {scores['random']:<15.2f} {scores['move_away']:<15.2f} {scores['move_closer']:<15.2f}")

        print("-" * 60)
    
    @staticmethod
    def move_away(main_boat: Boat, target_boats: List[Boat], map_grid: np.ndarray) -> List[BoatMoveDirection]:
        """
        Determines movement steps to move the main boat away from the nearest target boat.

        Args:
            main_boat (Boat): The boat to move.
            target_boats (List[Boat]): List of boats to evaluate the nearest target.
            map_grid (np.ndarray): The game map grid.

        Returns:
            List[BoatMoveDirection]: A list of movement steps to move away from the nearest boat.
        """
        nearest_boat = Strategy.find_nearest_boat(main_boat, target_boats)
        if not nearest_boat:
            return []  # No nearest boat, no movement needed

        current_x, current_y = main_boat.loc['x'], main_boat.loc['y']
        target_x, target_y = nearest_boat.loc['x'], nearest_boat.loc['y']
        valid_moves = []

        # Calculate movement directions to increase distance
        for _ in range(main_boat.mvdist):  # Restrict moves to boat's maximum movement distance
            potential_moves = []
            if current_x < target_x:
                potential_moves.append(BoatMoveDirection.UP)
            elif current_x > target_x:
                potential_moves.append(BoatMoveDirection.DOWN)

            if current_y < target_y:
                potential_moves.append(BoatMoveDirection.LEFT)
            elif current_y > target_y:
                potential_moves.append(BoatMoveDirection.RIGHT)

            # Choose the first valid move from potential_moves
            for move in potential_moves:
                new_x, new_y = current_x, current_y
                if move == BoatMoveDirection.UP:
                    new_x -= 1
                elif move == BoatMoveDirection.DOWN:
                    new_x += 1
                elif move == BoatMoveDirection.LEFT:
                    new_y -= 1
                elif move == BoatMoveDirection.RIGHT:
                    new_y += 1

                # Check if the position is valid and unoccupied
                if (
                    0 <= new_x < map_grid.shape[0] and
                    0 <= new_y < map_grid.shape[1] and
                    map_grid[new_x, new_y] is None
                ):
                    valid_moves.append(move)
                    current_x, current_y = new_x, new_y  # Update current position
                    break

        return valid_moves

    @staticmethod
    def move_closer(main_boat: Boat, target_boats: List[Boat], map_grid: np.ndarray) -> List[BoatMoveDirection]:
        """
        Determines movement steps to move the main boat closer to the nearest target boat.

        Args:
            main_boat (Boat): The boat to move.
            target_boats (List[Boat]): List of boats to evaluate the nearest target.
            map_grid (np.ndarray): The game map grid.

        Returns:
            List[BoatMoveDirection]: A list of movement steps to move closer to the nearest boat.
        """
        nearest_boat = Strategy.find_nearest_boat(main_boat, target_boats)
        if not nearest_boat:
            return []  # No nearest boat, no movement needed

        current_x, current_y = main_boat.loc['x'], main_boat.loc['y']
        target_x, target_y = nearest_boat.loc['x'], nearest_boat.loc['y']
        valid_moves = []

        # Calculate movement directions to decrease distance
        for _ in range(main_boat.mvdist):  # Restrict moves to boat's maximum movement distance
            potential_moves = []
            if current_x < target_x:
                potential_moves.append(BoatMoveDirection.DOWN)
            elif current_x > target_x:
                potential_moves.append(BoatMoveDirection.UP)

            if current_y < target_y:
                potential_moves.append(BoatMoveDirection.RIGHT)
            elif current_y > target_y:
                potential_moves.append(BoatMoveDirection.LEFT)

            # Choose the first valid move from potential_moves
            for move in potential_moves:
                new_x, new_y = current_x, current_y
                if move == BoatMoveDirection.UP:
                    new_x -= 1
                elif move == BoatMoveDirection.DOWN:
                    new_x += 1
                elif move == BoatMoveDirection.LEFT:
                    new_y -= 1
                elif move == BoatMoveDirection.RIGHT:
                    new_y += 1

                # Check if the position is valid and unoccupied
                if (
                    0 <= new_x < map_grid.shape[0] and
                    0 <= new_y < map_grid.shape[1] and
                    map_grid[new_x, new_y] is None
                ):
                    valid_moves.append(move)
                    current_x, current_y = new_x, new_y  # Update current position
                    break

        return valid_moves
    
    @staticmethod
    def arrange_all_moves(game: Game, user_id: str) -> Dict[Boat, List[BoatMoveDirection]]:
        """
        Arranges movement steps for all boats based on evaluation results.
        - Evaluates movement strategies for all boats.
        - Chooses the movement type (random, move closer, move away) with the highest score.
        - Generates movement steps for each boat accordingly.

        Args:
            game (Game): The game instance.
            user_id (str): The user ID (attacker or defender).

        Returns:
            Dict[Boat, List[BoatMoveDirection]]: A dictionary where each boat is a key,
            and the value is a list of movement directions.
        """
        # Generate evaluation results for all boats
        evaluation_results = Strategy.movement_evaluate(game, user_id)
        move_steps: Dict[Boat, List[BoatMoveDirection]] = {}

        # Determine ally and enemy boats
        if user_id == ATTACKER_ID:
            ally_boats = game.attacker_boats
            enemy_boats = game.defender_boats
        elif user_id == DEFENDER_ID:
            ally_boats = game.defender_boats
            enemy_boats = game.attacker_boats
        else:
            return {}

        for boat, scores in evaluation_results.items():
            # Determine the highest-scoring movement option
            best_option = max(scores, key=scores.get)

            # Generate movement steps based on the chosen option
            if best_option == "move_closer":
                moves = Strategy.move_closer(boat, enemy_boats, game.map_grid)
            elif best_option == "move_away":
                moves = Strategy.move_away(boat, ally_boats, game.map_grid)
            elif best_option == "random":
                moves = Strategy.move_randomly(boat, game.map_grid)
            else:
                moves = []

            # Store the generated movement steps
            move_steps[boat] = moves if moves else []

        return move_steps
    
    @staticmethod
    def show_moves(arranged_moves: Dict[Boat, List[BoatMoveDirection]]):
        """
        Displays the arranged moves for all boats based on the movement strategy.

        Args:
            arranged_moves (Dict[Boat, List[BoatMoveDirection]]): A dictionary where each Boat is a key,
                and the value is a list of BoatMoveDirection representing its moves.
        """
        print("\nArranged Moves:")
        print(f"{'Boat Name':<15} {'Moves':<50}")
        print("-" * 65)

        for boat, move_list in arranged_moves.items():
            move_str = ', '.join([move.name for move in move_list]) if move_list else "No moves"
            print(f"{boat.name:<15} {move_str:<50}")

        print("-" * 65)
    
    @staticmethod
    def score_move_limit(boat: Boat, map_grid: np.ndarray) -> float:
        """
        Scores the enemy boat's movement restriction based on the proportion of 
        valid positions it can move to compared to its theoretical maximum.

        Args:
            boat (Boat): The enemy boat to evaluate.
            map_grid (np.ndarray): The game map grid.

        Returns:
            float: A score between 0 and 10, where higher scores indicate more restriction.
        """
        
        base_moves_block = 9 
        
        valid_positions = Strategy.get_valid_positions(boat, map_grid)
        original_moves_count = len(boat.get_move_range())
        valid_moves_count = len(valid_positions)
        
        if base_moves_block >= valid_moves_count:
            return 10
        else:
            restriction_ratio =  (original_moves_count - valid_moves_count) / (original_moves_count - base_moves_block)
            # Normalize to 0-10 scale
            normalized_score = restriction_ratio * 10
            return round(normalized_score, 2)
            
    @staticmethod
    def score_self_hp(boat: Boat, map_grid: np.ndarray) -> float:
        """
        Scores the boat's health inversely, where lower health results in a higher score.

        Args:
            boat (Boat): The boat to evaluate.
            map_grid (np.ndarray): The game map grid (not used in this calculation, but kept for consistency).

        Returns:
            float: A score between 0 and 10, where higher scores indicate lower health.
        """
        # Theoretical maximum health points for a boat
        if boat.boat_size == BoatSize.LARGE:
            max_hp = 3  
        elif boat.boat_size == BoatSize.MEDIUM:
            max_hp = 2  
        elif boat.boat_size == BoatSize.SMALL:
            max_hp = 1
        else:
            return 0.0
       
        # Calculate inverse health score
        inverse_hp_score = max_hp - boat.hp
        normalized_score = (inverse_hp_score / max_hp) * 10

        # Ensure the score is rounded to 2 decimal places
        return round(normalized_score, 2)
    
    @staticmethod
    def score_target(boat: Boat, map_grid: np.array, attack_records: List[Dict[str, Any]] = None) -> float:
        """
        Scores the boat based on how many times it has been targeted by attacks.
        The score considers the overlap of attack ranges with the boat's move range.
        Lower target counts result in a higher score.

        Args:
            boat (Boat): The boat to evaluate.
            attack_records (List[Dict[str, Any]]): List of attack records. Each record contains:
                - 'loc': {'x': int, 'y': int} (attack center location)
                - 'weapon_type': WeaponType (type of weapon used in the attack)

        Returns:
            float: A score between 0 and 10, where lower target counts result in higher scores.
        """
        if not attack_records:
            return 10.0  # No attacks recorded, perfect score

        been_target_count = 0

        # Get the boat's movement range positions
        move_range_positions = boat.get_move_range()
        move_range_positions = {
            (boat.loc['x'] + dx, boat.loc['y'] + dy)
            for dx, dy in move_range_positions
            if 0 <= boat.loc['x'] + dx < len(map_grid) and 0 <= boat.loc['y'] + dy < len(map_grid[0])
        }

        # Analyze each attack record
        for record in attack_records:
            attack_x, attack_y = record['loc']['x'], record['loc']['y']
            attack_range = boat.get_attack_range(record['weapon_type'])

            # Check how many attack positions overlap with the boat's movement range
            overlap_count = sum(
                1 for dx, dy in attack_range
                if (attack_x + dx, attack_y + dy) in move_range_positions
            )

            # If more than half the attack range overlaps with the boat's move range, count it as a target
            if overlap_count >= len(attack_range) / 2:
                been_target_count += 1

        # Calculate the score based on the number of times targeted and HP
        normalized_score = max(0, 10 - (been_target_count / boat.hp) * 10)
        return round(normalized_score, 2)
    
    @staticmethod
    def score_survival(boats: List[Boat], user_id: str) -> float:
        """
        Scores the survival rate of boats based on the proportion of boats still alive compared to the initial total.
        A higher survival rate results in a higher score.

        Args:
            boats (List[Boat]): The list of boats to evaluate (belonging to the given user ID).
            user_id (str): The user ID, either ATTACKER_ID or DEFENDER_ID, to determine the base number of boats.

        Returns:
            float: A score between 0 and 10, where a higher score indicates a higher survival rate.
        """
        # Determine the base number of boats based on the user ID
        base_boat_count = 8 if user_id == ATTACKER_ID else 9

        # Count the number of boats still alive
        alive_boats = len(boats)

        # Calculate survival ratio
        survival_ratio = alive_boats / base_boat_count

        # Normalize the score to 0-10
        score = survival_ratio * 10
        return round(score, 2)
    
    @staticmethod
    def attack_evaluate(game: Game, user_id: str) -> Dict[Boat, Dict[str, float]]:
        """
        Evaluates the attack strategy for boats based on scoring criteria and weights.

        Attack Options:
        - Direct attack
        - Average attack
        - Focused attack

        Scoring Criteria:
        - Criterion 1: Enemy movement restriction (higher restriction = higher score)
        - Criterion 2: Enemy HP (lower HP = higher score)
        - Criterion 3: Enemy target count (lower target count = higher score)
        - Criterion 4: Enemy survival rate (higher survival rate = higher score)

        Weight Table:
            Attack/Criteria     Criterion1 Weight  Criterion2 Weight  Criterion3 Weight  Criterion4 Weight
            Direct Attack       0.4                0.3                0.2                0.1
            Average Attack      0.1                0.2                0.3                0.4
            Focused Attack      0.2                0.1                0.4                0.3

        Args:
            game (Game): The game instance.
            user_id (str): The user ID (attacker or defender).
            attack_records (List[Dict[str, Any]]): List of attack records for scoring.

        Returns:
            Dict[Boat, Dict[str, float]]: A dictionary containing the evaluated scores for each attack option for all boats.
        """
        # Determine ally and enemy boats based on the user ID
        if user_id == ATTACKER_ID:
            ally_boats = game.attacker_boats
            enemy_boats = game.defender_boats
        elif user_id == DEFENDER_ID:
            ally_boats = game.defender_boats
            enemy_boats = game.attacker_boats
        else:
            return {}

        # Weight configuration
        weights = {
            "direct": [0.3, 0.1, 0.2, 0.4],
            "average": [0.1, 0.2, 0.3, 0.4],
            "focused": [0.2, 0.1, 0.4, 0.3]
        }

        evaluation_results: Dict[Boat, Dict[str, float]] = {}
        # attack_records: List[Dict[str, Any]] = None
        attack_records = None

        for enemy_boat in enemy_boats:
            # Calculate scores for each criterion
            score_move_limit_val = Strategy.score_move_limit(enemy_boat, game.map_grid)
            score_hp_val = Strategy.score_self_hp(enemy_boat, game.map_grid)
            score_target_val = Strategy.score_target(enemy_boat, game.map_grid, attack_records)
            score_survival_val = Strategy.score_survival(enemy_boats, user_id)

            # Attack option evaluations
            scores = {}
            for option, weight in weights.items():
                total_score = (
                    weight[0] * score_move_limit_val +
                    weight[1] * score_hp_val +
                    weight[2] * score_target_val +
                    weight[3] * score_survival_val
                )
                scores[option] = round(total_score, 2)

            evaluation_results[enemy_boat] = scores

        return evaluation_results

    @staticmethod
    def show_attack_evaluation_results(attack_evaluation_results: Dict[Boat, Dict[str, float]]):
        """
        Displays the attack evaluation results for each boat.

        Args:
            attack_evaluation_results (Dict[Boat, Dict[str, float]]): A dictionary where each Boat is a key,
                and the value is another dictionary containing scores for each attack option.
        """
        print("\nAttack Evaluation Results:")
        print(f"{'Boat Name':<20} {'Direct Attack':<15} {'Average Attack':<15} {'Focused Attack':<15}")
        print("-" * 65)

        for boat, scores in attack_evaluation_results.items():
            print(f"{boat.name:<20} {scores['direct']:<15.2f} {scores['average']:<15.2f} {scores['focused']:<15.2f}")

        print("-" * 65)
    
    @staticmethod
    def attack_average(enemy_boat: Boat, map_grid: np.ndarray) -> List[Dict[str, Any]]:
        """
        Determines the most effective attack group for an enemy boat by comparing
        attack ranges for two predefined groups and adjusts the target to avoid friendly fire.

        Args:
            enemy_boat (Boat): The enemy boat being attacked.
            map_grid (np.ndarray): The game map grid.

        Returns:
            List[Dict[str, Any]]: A list of adjusted attack positions with their weapon types.
        """
        x, y = enemy_boat.loc['x'], enemy_boat.loc['y']
        
        # Define attack groups
        attack_group1 = [
            {'loc': {'x': x, 'y': y - 1}, 'weapon_type': WeaponType.M_SONIC},
            {'loc': {'x': x, 'y': y + 2}, 'weapon_type': WeaponType.M_SONIC}
        ]
        attack_group2 = [
            {'loc': {'x': x + 1, 'y': y}, 'weapon_type': WeaponType.M_SONIC},
            {'loc': {'x': x - 2, 'y': y}, 'weapon_type': WeaponType.M_SONIC}
        ]
        
        def calculate_effectiveness(group):
            """Calculate the number of enemy boats covered by the attack range of a group."""
            effectiveness = 0
            for step in group:
                attack_x, attack_y = step['loc']['x'], step['loc']['y']
                attack_range = enemy_boat.get_attack_range(step['weapon_type'])
                
                for dx, dy in attack_range:
                    target_x, target_y = attack_x + dx, attack_y + dy
                    if (0 <= target_x < map_grid.shape[0] and
                            0 <= target_y < map_grid.shape[1]):
                        target = map_grid[target_x, target_y]
                        if target is not None and target.owner_id != enemy_boat.owner_id:  # Count enemy boats only
                            effectiveness += 1
            return effectiveness

        # Evaluate effectiveness for both groups
        effectiveness_group1 = calculate_effectiveness(attack_group1)
        effectiveness_group2 = calculate_effectiveness(attack_group2)

        # Select the most effective group
        best_group = attack_group1 if effectiveness_group1 > effectiveness_group2 else attack_group2

        # Adjust each target position in the selected group to avoid friendly fire
        adjusted_targets = []
        for step in best_group:
            adjusted_step = Strategy.adjust_attack_position(enemy_boat, step, map_grid)
            if adjusted_step:
                adjusted_targets.append(adjusted_step)
        
        # Return the adjusted attack positions
        return adjusted_targets

    @staticmethod
    def attack_concentrated(enemy_boat: Boat, map_grid: np.ndarray) -> List[Dict[str, Any]]:
        """
        Determines the most effective attack positions for concentrated attacks,
        prioritizing areas with the highest number of enemy boats, and executes two attacks.

        Args:
            enemy_boat (Boat): The enemy boat being attacked.
            map_grid (np.ndarray): The game map grid.

        Returns:
            List[Dict[str, Any]]: A list of two attack positions with their weapon types.
        """
        x, y = enemy_boat.loc['x'], enemy_boat.loc['y']
        potential_targets = []
        
        # Get attack range for each potential target location
        for dx in range(-1, 2):  # Consider surrounding locations in a 3x3 grid
            for dy in range(-1, 2):
                target_x, target_y = x + dx, y + dy
                if (0 <= target_x < map_grid.shape[0] and
                        0 <= target_y < map_grid.shape[1]):
                    attack_range = enemy_boat.get_attack_range(WeaponType.M_SONIC)
                    enemy_count = 0
                    
                    # Count the number of enemy boats in this attack range
                    for ax, ay in attack_range:
                        range_x, range_y = target_x + ax, target_y + ay
                        if (0 <= range_x < map_grid.shape[0] and
                                0 <= range_y < map_grid.shape[1]):
                            target = map_grid[range_x, range_y]
                            if target is not None and target.owner_id != enemy_boat.owner_id:
                                enemy_count += 1
                    
                    # Record potential target
                    potential_targets.append({
                        'loc': {'x': target_x, 'y': target_y},
                        'weapon_type': WeaponType.M_SONIC,
                        'enemy_count': enemy_count
                    })
        
        # Sort potential targets by the number of enemies affected, descending
        potential_targets.sort(key=lambda t: t['enemy_count'], reverse=True)

        # Select the top two targets
        selected_targets = potential_targets[:2]

        # Adjust the positions to avoid friendly fire
        adjusted_attacks = []
        for target in selected_targets:
            adjusted_target = Strategy.adjust_attack_position(enemy_boat, target, map_grid)
            if adjusted_target:
                adjusted_attacks.append(adjusted_target)

        return adjusted_attacks
    
    @staticmethod
    def arrange_all_attack(game: Game, user_id: str) -> Optional[Dict[Boat, Optional[Dict[str, Any]]]]:
        """
        Arranges attack strategies for all ally boats by assigning each attack to the boat
        with the lowest HP first, ensuring no boat attacks more than once.

        Args:
            game (Game): The game instance.
            user_id (str): The ID of the user (attacker or defender).

        Returns:
            Dict[Boat, Optional[Dict[str, Any]]]: A dictionary where each ally boat is a key,
                and the value is its assigned attack (if any), or None. If any issue occurs, returns None.
        """
        # Determine ally and enemy boats based on the user ID
        if user_id == ATTACKER_ID:
            ally_boats = game.attacker_boats
            enemy_boats = game.defender_boats
        elif user_id == DEFENDER_ID:
            ally_boats = game.defender_boats
            enemy_boats = game.attacker_boats
        else:
            return None  # Invalid user_id

        # No boats to work with
        if not ally_boats or not enemy_boats:
            return None

        try:
            # Evaluate attack options for all enemy boats
            attack_evaluation = Strategy.attack_evaluate(game, user_id)
            if not attack_evaluation:
                return None  # Evaluation failed

            Strategy.show_attack_evaluation_results(attack_evaluation)

            # Sort enemy boats by lowest HP first
            sorted_enemy_boats = sorted(enemy_boats, key=lambda b: b.hp)
            attack_list = []

            # Generate attack actions for each enemy boat
            for enemy_boat in sorted_enemy_boats:
                scores = attack_evaluation.get(enemy_boat)
                if not scores:
                    return None  # Missing evaluation scores

                if scores["direct"] >= max(scores["average"], scores["focused"]):
                    # Direct attack returns a single attack or None
                    attack = Strategy.attack_directly(enemy_boat, ally_boats, game.map_grid)
                    if attack:
                        attack_list.append(attack)
                elif scores["average"] >= max(scores["direct"], scores["focused"]):
                    # Append all attacks from attack_average
                    attacks = Strategy.attack_average(enemy_boat, game.map_grid)
                    if not attacks:
                        return None  # Invalid average attack results
                    attack_list.extend(attacks)
                else:
                    # Append all attacks from attack_concentrated
                    attacks = Strategy.attack_concentrated(enemy_boat, game.map_grid)
                    if not attacks:
                        return None  # Invalid concentrated attack results
                    attack_list.extend(attacks)

            # Ensure attack_list is valid
            if not attack_list:
                return None

            assigned_attacks = {}

            # Assign attacks to ally boats
            for boat in ally_boats:
                for attack in attack_list[:]:  # Iterate over a copy of attack_list
                    if not attack or 'weapon_type' not in attack or 'loc' not in attack:
                        return None  # Invalid attack format

                    weapon_type = attack['weapon_type']

                    # Check if the boat has the required weapon
                    if weapon_type == WeaponType.M_SONIC and boat.sonic_missile_count > 0:
                        assigned_attacks[boat] = attack
                        boat.sonic_missile_count -= 1
                        attack_list.remove(attack)
                        break
                    elif weapon_type == WeaponType.M_HYPERSONIC and boat.hypersonic_missile_count > 0:
                        assigned_attacks[boat] = attack
                        boat.hypersonic_missile_count -= 1
                        attack_list.remove(attack)
                        break
                    else:
                        # Attempt to switch weapons if the preferred one isn't available
                        if boat.sonic_missile_count > 0:
                            attack['weapon_type'] = WeaponType.M_SONIC
                            assigned_attacks[boat] = attack
                            boat.sonic_missile_count -= 1
                            attack_list.remove(attack)
                            break
                        elif boat.hypersonic_missile_count > 0:
                            attack['weapon_type'] = WeaponType.M_HYPERSONIC
                            assigned_attacks[boat] = attack
                            boat.hypersonic_missile_count -= 1
                            attack_list.remove(attack)
                            break

                # If no attack could be assigned, set to None
                if boat not in assigned_attacks:
                    assigned_attacks[boat] = None

            return assigned_attacks

        except Exception as e:
            print(f"Error in arrange_all_attack: {e}")
            return None