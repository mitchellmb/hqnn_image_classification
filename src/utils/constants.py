leaves_target_dict = {
    0: 'Mango (P0) healthy',
    1: 'Arjun (P1) healthy',
    2: 'Alstonia Scholaris (P2) healthy',
    3: 'Gauva (P3) healthy',
    4: 'Jamun (P5) healthy',
    5: 'Jatropha (P6) healthy',
    6: 'Pongamia Pinnata (P7) healthy',
    7: 'Basil (P8) healthy', # no diseased match
    8: 'Pomegranate (P9) healthy',
    9: 'Lemon (P10) healthy',
    10: 'Chinar (P11) healthy',

    11: 'Mango (P0) diseased',
    12: 'Arjun (P1) diseased',
    13: 'Alstonia Scholaris (P2) diseased',
    14: 'Gauva (P3) diseased',
    15: 'Bael (P4) diseased', # no healthy match
    16: 'Jamun (P5) diseased',
    17: 'Jatropha (P6) diseased',
    18: 'Pongamia Pinnata (P7) diseased',
    19: 'Pomegranate (P9) diseased',
    20: 'Lemon (P10) diseased',
    21: 'Chinar (P11) diseased'
    }

combine_healthy_unhealthy_dict = {
    #Healthy labels mapping
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: -1,
    8: 7,
    9: 8,
    10: 9,
    
    #Unhealthy labels mapping
    11: 0,
    12: 1,
    13: 2,
    14: 3,
    15: -1,
    16: 4,
    17: 5,
    18: 6,
    19: 7,
    20: 8,
    21: 9
}