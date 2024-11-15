from config import MAP_SIZE, ATTACKER_ID, DEFENDER_ID, ROUND
from strategy import Strategy
from game import Game
from typing import Dict

def initial_game()->Game:
    # Create Map
    game = Game(map_size=MAP_SIZE)
    game.create_map()

    # Set boat name
    attacker_names = ["Attacker_1", "Attacker_2", "Attacker_3", "Attacker_4", "Attacker_5", "Attacker_6", "Attacker_7", "Attacker_8"]
    defender_names = ["Defender_1", "Defender_2", "Defender_3", "Defender_4", "Defender_5", "Defender_6", "Defender_7", "Defender_8", "Defender_9"]
    
    # Create boats
    attacker_boats = game.create_attacker(ATTACKER_ID, attacker_names)
    defender_boats = game.create_defender(DEFENDER_ID, defender_names)

    # Set boat location (attacker at top, defender at bottom, all sperate)
    attacker_coordinates = Strategy.place_widest(attacker_boats, is_attacker=True)
    defender_coordinates = Strategy.place_widest(defender_boats, is_attacker=False)

    # Place boats location
    game.place_all_boats(attacker_boats, coordinates=attacker_coordinates)
    game.place_all_boats(defender_boats, coordinates=defender_coordinates)

    # Show placement
    game.print_map_grid()

    return game

def start_game()->Dict[str, int]:
    # Initialize the game
    game = initial_game()
    round = ROUND

    while round > 0:
        
        # < Attacker Atteck>
        # Generate attack steps for attacker boats
        attack_steps =  Strategy.arrange_all_attack(game, ATTACKER_ID)
        
        # < Defender Move>
        # Generate movement steps for defender boats
        move_steps = Strategy.arrange_all_moves(game, DEFENDER_ID)
        
        # Excute process
        game.round(move_steps=move_steps, attack_steps=attack_steps)

        # < Defender attack>
        # Generate attack steps for defender boats
        attack_steps = Strategy.arrange_all_attack(game, DEFENDER_ID)
        
        # < Attacker Move>
        # Generate movement steps for attacker boats
        move_steps =  Strategy.arrange_all_moves(game, ATTACKER_ID)
        
        # Excute process
        game.round(move_steps=move_steps, attack_steps=attack_steps)
        game.print_map_grid()
        
        print(f"Attacker boats: {len(game.attacker_boats)}, Defender boats: {len(game.defender_boats)}")

        round -=1
    
    result = {
        ATTACKER_ID : len(game.attacker_boats), 
        DEFENDER_ID : len(game.defender_boats)
    }
    return result 

if __name__ == "__main__":
    play_times = 50
    win_record = {
        ATTACKER_ID : 0,
        DEFENDER_ID : 0
    }
    for i in range(play_times):
        result = start_game()
        if result[ATTACKER_ID] > result[DEFENDER_ID ]:
            win_record[ATTACKER_ID] += 1
        elif result[DEFENDER_ID] > result[ATTACKER_ID]:
            win_record[DEFENDER_ID] += 1
        else:
            win_record[DEFENDER_ID] += 1
            win_record[ATTACKER_ID] += 1
    print(win_record)