# main/context_processors/facts.py

# Define your agricultural facts list here, matching the one used in your original index.html script.
AGRI_FACTS = [
    "India is the world's largest producer of milk, pulses, and jute.",
    "The monsoon is often called the 'true finance minister of India'.",
    "India's agriculture sector employs nearly half of the country's workforce.",
    "The staple crop Rice covers the largest area under cultivation in India.",
    "The Green Revolution in India started in the 1960s.",
    "India has 15 different agro-climatic zones.",
    "Black soil (Regur soil) is ideal for growing cotton and sugarcane.",
]

def global_facts(request):
    """
    Returns a dictionary of agricultural facts accessible globally in all templates.
    """
    # Note: We return the list of facts and choose one random fact for use in a template, 
    # but the simplest way to fix the ImportError is just to define the function.
    
    return {
        'GLOBAL_AGRI_FACTS_LIST': AGRI_FACTS,
    }