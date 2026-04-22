"""
Ground truth evaluation questions for O'Connors IMS RAG chatbot.

30 questions across 6 categories (5 each):
    - Policies: company-wide policies
    - Procedures: operational and safety procedures
    - SOPs: standard operating procedures for HVAC work
    - SWMSs: safe work method statements
    - Australian Standards: AS/NZS references
    - Cross-document: questions requiring synthesis from multiple sources

Each question includes:
    - question: the natural-language query
    - expected_sources: file name patterns that should appear in results
    - category: which IMS category it tests
    - key_terms: terms that must appear in a faithful answer
"""


EVAL_QUESTIONS = [
    # === Policies (5) ===
    {
        "question": "What is O'Connors' policy on drug and alcohol use in the workplace?",
        "expected_sources": ["DP01", "Drug and Alcohol"],
        "category": "Policies",
        "key_terms": ["drug", "alcohol", "testing", "workplace"],
    },
    {
        "question": "What are the key principles of the O'Connors Code of Conduct?",
        "expected_sources": ["CCP01", "Code of Conduct"],
        "category": "Policies",
        "key_terms": ["conduct", "behaviour", "integrity"],
    },
    {
        "question": "What is the O'Connors respectful workplace policy?",
        "expected_sources": ["RWP", "Respectful Workplace"],
        "category": "Policies",
        "key_terms": ["respectful", "harassment", "bullying", "workplace"],
    },
    {
        "question": "What does O'Connors' rehabilitation procedure cover?",
        "expected_sources": ["RP 01", "Rehabilitation"],
        "category": "Policies",
        "key_terms": ["rehabilitation", "return to work", "injury"],
    },
    {
        "question": "What is the psychosocial safety procedure at O'Connors?",
        "expected_sources": ["PSP01", "Psychosocial"],
        "category": "Policies",
        "key_terms": ["psychosocial", "mental health", "safety"],
    },

    # === Procedures (5) ===
    {
        "question": "What is the after-hours call out procedure?",
        "expected_sources": ["AHP01", "After Hours Call Out"],
        "category": "Procedures",
        "key_terms": ["after hours", "call out", "emergency"],
    },
    {
        "question": "What is the emergency plan for the head office and workshops?",
        "expected_sources": ["EP01", "Emergency Plan"],
        "category": "Procedures",
        "key_terms": ["emergency", "evacuation", "assembly"],
    },
    {
        "question": "What is the isolation and lock out procedure?",
        "expected_sources": ["WP05", "Isolation", "Lock Out"],
        "category": "Procedures",
        "key_terms": ["isolation", "lock out", "tag", "energy"],
    },
    {
        "question": "What is the first aid procedure at O'Connors?",
        "expected_sources": ["FP 01", "First Aid"],
        "category": "Procedures",
        "key_terms": ["first aid", "injury", "treatment"],
    },
    {
        "question": "How does O'Connors handle contract management?",
        "expected_sources": ["CP02", "SP03", "Contract Management"],
        "category": "Procedures",
        "key_terms": ["contract", "management", "variation"],
    },

    # === SOPs (5) ===
    {
        "question": "What is the procedure for handling refrigerant gases?",
        "expected_sources": ["SOP", "refrigerant"],
        "category": "SOPs",
        "key_terms": ["refrigerant", "gas", "handling", "PPE"],
    },
    {
        "question": "What are the requirements for working with electrical equipment according to O'Connors SOPs?",
        "expected_sources": ["SOP", "electrical"],
        "category": "SOPs",
        "key_terms": ["electrical", "isolation", "safety"],
    },
    {
        "question": "What is SOP for chiller maintenance?",
        "expected_sources": ["SOP", "chiller"],
        "category": "SOPs",
        "key_terms": ["chiller", "maintenance", "inspection"],
    },
    {
        "question": "What PPE is required for HVAC maintenance work?",
        "expected_sources": ["SOP", "PPE"],
        "category": "SOPs",
        "key_terms": ["PPE", "gloves", "safety glasses", "protection"],
    },
    {
        "question": "What is the procedure for compliance and statutory testing?",
        "expected_sources": ["SP06", "Compliance", "Statutory"],
        "category": "SOPs",
        "key_terms": ["compliance", "statutory", "testing", "inspection"],
    },

    # === SWMSs (5) ===
    {
        "question": "What are the safe work procedures for working at heights?",
        "expected_sources": ["SWMS", "height", "WP03"],
        "category": "SWMSs",
        "key_terms": ["height", "fall", "harness", "anchor"],
    },
    {
        "question": "What is the SWMS for hot work activities?",
        "expected_sources": ["SWMS", "hot work"],
        "category": "SWMSs",
        "key_terms": ["hot work", "welding", "fire", "permit"],
    },
    {
        "question": "What are the hazards identified for working in confined spaces?",
        "expected_sources": ["SWMS", "confined space"],
        "category": "SWMSs",
        "key_terms": ["confined space", "atmosphere", "ventilation", "rescue"],
    },
    {
        "question": "What safe work controls are required for manual handling tasks?",
        "expected_sources": ["SWMS", "manual handling", "WP04"],
        "category": "SWMSs",
        "key_terms": ["manual handling", "lifting", "ergonomic"],
    },
    {
        "question": "What is the SWMS for silica duct removal?",
        "expected_sources": ["SWMS", "Silica Duct"],
        "category": "SWMSs",
        "key_terms": ["silica", "duct", "removal", "respiratory"],
    },

    # === Australian Standards (5) ===
    {
        "question": "What does AS 1668 specify about ventilation and air conditioning?",
        "expected_sources": ["AS 1668", "ventilation"],
        "category": "Standards",
        "key_terms": ["ventilation", "air conditioning", "building"],
    },
    {
        "question": "What are the requirements for fire detection systems under Australian Standards?",
        "expected_sources": ["AS 1670", "fire detection"],
        "category": "Standards",
        "key_terms": ["fire", "detection", "alarm", "system"],
    },
    {
        "question": "What does AS 3666 require for air handling and water systems maintenance?",
        "expected_sources": ["AS 3666", "air handling", "Legionella"],
        "category": "Standards",
        "key_terms": ["air handling", "water", "Legionella", "maintenance"],
    },
    {
        "question": "What are the acoustic requirements specified in AS 1055?",
        "expected_sources": ["AS 1055", "Acoustics"],
        "category": "Standards",
        "key_terms": ["acoustic", "noise", "measurement"],
    },
    {
        "question": "What does AS 1807 specify about biological safety cabinets?",
        "expected_sources": ["AS 1807", "biological", "safety cabinet"],
        "category": "Standards",
        "key_terms": ["biological", "safety cabinet", "cytotoxic", "test"],
    },

    # === Cross-document (5) ===
    {
        "question": "What safety requirements apply to both electrical work and confined space entry?",
        "expected_sources": ["SOP", "SWMS", "electrical", "confined"],
        "category": "Cross-doc",
        "key_terms": ["isolation", "permit", "ventilation", "electrical"],
    },
    {
        "question": "How do O'Connors procedures and Australian Standards together address HVAC maintenance?",
        "expected_sources": ["SOP", "AS", "HVAC", "maintenance"],
        "category": "Cross-doc",
        "key_terms": ["maintenance", "HVAC", "standard", "procedure"],
    },
    {
        "question": "What documentation and forms are required when reporting a workplace incident?",
        "expected_sources": ["Procedure", "Form", "incident"],
        "category": "Cross-doc",
        "key_terms": ["incident", "report", "form", "investigation"],
    },
    {
        "question": "What are the PPE requirements across different SWMS documents?",
        "expected_sources": ["SWMS", "PPE"],
        "category": "Cross-doc",
        "key_terms": ["PPE", "gloves", "helmet", "protection"],
    },
    {
        "question": "How do O'Connors service procedures relate to commissioning procedures?",
        "expected_sources": ["SP09", "SP07", "CP10", "Commissioning"],
        "category": "Cross-doc",
        "key_terms": ["service", "commissioning", "handover"],
    },
]


def get_questions_by_category(category: str = None) -> list[dict]:
    """Return questions filtered by category, or all if None."""
    if category is None:
        return EVAL_QUESTIONS
    return [q for q in EVAL_QUESTIONS if q["category"] == category]


def get_categories() -> list[str]:
    """Return all unique categories."""
    return sorted(set(q["category"] for q in EVAL_QUESTIONS))
