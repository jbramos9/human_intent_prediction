HIPHOP = "hiphop"
VIDEOS = "videos"
INTENT = "intent"
GAZE_SEQ = "gaze_seq"

GAZED_OBJECT_MAPPING = {
    "none": 0.0,
    "Bag": 1.0,
    "Book": 2.0,
    "Bottle": 3.0,
    "Bowl": 4.0,
    "Broom": 5.0,
    "Chair": 6.0,
    "Cup": 7.0,
    "Fruits": 8.0,
    "Laptop": 9.0,
    "Pillow": 10.0,
    "Racket": 11.0,
    "Rug": 12.0,
    "Sandwich": 13.0,
    "Umbrella": 14.0,
    "Utensils": 15.0,
}
GAZED_OBJECT_MAPPING_R = {
    0.0 : "none",
    1.0 : "Bag",
    2.0 : "Book",
    3.0 : "Bottle",
    4.0 : "Bowl",
    5.0 : "Broom",
    6.0 : "Chair",
    7.0 : "Cup",
    8.0 : "Fruits",
    9.0 : "Laptop",
    10. : "Pillow",
    11. : "Racket",
    12. : "Rug",
    13. : "Sandwich",
    14. : "Umbrella",
    15. : "Utensils",
}

INTENTIONS_MAPPING = {
    "Indeterminate" : 0,
    "Clean the Area" : 1,
    "Drink" : 2,
    "Eat" : 3,
    "Go Outside" : 4,
    "Rest" : 5,
    "Study" : 6,
    "Spontaneous" : 7,
}
INTENTIONS_MAPPING_R = {
    0: "Indeterminate",
    1: "Clean the Area",
    2: "Drink",
    3: "Eat",
    4: "Go Outside",
    5: "Rest",
    6: "Study",
    7: "Spontaneous",
}