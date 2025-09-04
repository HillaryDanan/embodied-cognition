"""
Test if LLMs detect basic physics violations that infants recognize
Based on Baillargeon (2004) and Spelke et al. (1992)
"""

physics_tests = {
    'object_permanence': {
        'normal': "The ball rolled behind the screen and remained there",
        'violation': "The ball rolled behind the screen and ceased to exist"
    },
    'gravity': {
        'normal': "The apple fell from the tree to the ground",
        'violation': "The apple fell from the tree toward the sky"
    },
    'support': {
        'normal': "The book rested on the table",
        'violation': "The book floated in midair without support"
    }
}

# TODO: Implement measurement using retroactive update metric
