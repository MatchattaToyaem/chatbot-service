"""
Vincent's 45 Standard Questions for O'Connors IMS RAG Evaluation.

Categories:
    Q01-Q15: MSDS / Chemical Safety (Material Safety Data Sheets)
    Q16-Q30: Company Policies (HR, WHS, operational policies)
    Q31-Q45: Australian Standards / Technical Compliance

Each question includes:
    - qid: Question identifier
    - category: One of "MSDS", "Policy", "Standards"
    - question: The natural-language question
    - key_terms: Expected keywords/phrases that a correct answer should contain
    - source_hint: Expected document or section reference
"""

VINCENT_QUESTIONS = [
    # =========================================================================
    # Q01-Q15: MSDS / Chemical Safety
    # =========================================================================
    {
        "qid": "V01",
        "category": "MSDS",
        "question": "I'm currently connecting a cylinder of Gas2Go R134A to our lower pressure piping system. What specific device should I use, and what should I do if the valve resists opening?",
        "key_terms": ["pressure reducing regulator", "lower pressure", "100 psig", "do not apply extra leverage", "contact your supervisor"],
        "source_hint": "Gas2Go R134A SDS Section 7",
    },
    {
        "qid": "V02",
        "category": "MSDS",
        "question": "I've just spilled a large amount of Loctite 406 Wicking Grade Instant Adhesive on the workshop floor, and I'm holding a cloth to mop it up. Is this the right approach?",
        "key_terms": ["do not use cloths", "exothermic polymerization", "smoke", "irritating vapors", "flood with water", "scrape"],
        "source_hint": "Loctite 406 SDS Section 6",
    },
    {
        "qid": "V03",
        "category": "MSDS",
        "question": "I'm at the storage facility and notice a small fire near our LPG cylinders. They look like they might be getting hot. Should I grab a hose and approach them to cool them down?",
        "key_terms": ["do not approach", "rupture", "pressure relief", "cool", "water", "protected", "safe location"],
        "source_hint": "LPG SDS Section 5.3",
    },
    {
        "qid": "V04",
        "category": "MSDS",
        "question": "I'm organising the chemical storage unit and placing Diggers Mineral Turpentine next to some pool chlorine and nitrates to save space. Is there any risk with this setup?",
        "key_terms": ["severe risk", "oxidising agents", "nitrates", "chlorine", "violently", "explosively", "ignition"],
        "source_hint": "Mineral Turpentine SDS Section 5",
    },
    {
        "qid": "V05",
        "category": "MSDS",
        "question": "I am about to saw through some Thermobreak No-Clad sheet for our HVAC ducting. What specific PPE do I need to wear to protect myself from the foil reinforcement layer?",
        "key_terms": ["dust mask", "P1", "glass fibres", "safety glasses", "gloves", "long-sleeved", "skin irritation"],
        "source_hint": "Thermobreak No-Clad SDS Section 8",
    },
    {
        "qid": "V06",
        "category": "MSDS",
        "question": "I've been working with Soudal PVC Pipe Cement Type N Clear in a deep trench and suddenly feel very dizzy and drowsy. What hazard is causing this, and what should my coworker do?",
        "key_terms": ["narcotic", "target organ toxicity", "drowsiness", "dizziness", "remove from exposure", "loosen clothing", "medical advice"],
        "source_hint": "Soudal PVC Cement SDS Section 2/4",
    },
    {
        "qid": "V07",
        "category": "MSDS",
        "question": "I am doing some manual soldering with RS 60/40 Sn-Pb wire inside a fairly enclosed tank. What is the limit for the fume concentration inside my welding helmet, and what happens if I breathe in too much?",
        "key_terms": ["5 mg/m3", "metal fume fever", "ventilation", "particulate"],
        "source_hint": "RS 60/40 Sn-Pb SDS Section 8",
    },
    {
        "qid": "V08",
        "category": "MSDS",
        "question": "I'm mixing Selleys Araldite Ultra Clear Part B and some just splashed directly onto my skin, causing a burning sensation. What immediate first aid must I apply?",
        "key_terms": ["remove contaminated clothing", "flush", "running water", "skin burns", "dry dressing", "blisters", "medical assistance"],
        "source_hint": "Araldite Ultra Clear SDS Section 4",
    },
    {
        "qid": "V09",
        "category": "MSDS",
        "question": "I'm about to transfer Unleaded Petrol into a large storage tank using a pump. How long do I need to wait after filling before I open the hatches, and why?",
        "key_terms": ["30 minutes", "static accumulator", "electrostatic charge", "explosion", "ignition", "grounding"],
        "source_hint": "Unleaded Petrol SDS Section 7",
    },
    {
        "qid": "V10",
        "category": "MSDS",
        "question": "I'm working with Blackwoods Stag Jointing Paste, which contains confirmed carcinogens. What specific procedures must I follow for my protective clothing when I leave the regulated area?",
        "key_terms": ["remove", "protective clothing", "point of exit", "impervious containers", "decontamination", "disposal", "shower"],
        "source_hint": "Stag Jointing Paste SDS Section 8",
    },
    {
        "qid": "V11",
        "category": "MSDS",
        "question": "I'm using Dy-Mark Spray & Mark aerosol inside a confined basement room with no ventilation. Is this safe to do if I'm just doing a quick mark?",
        "key_terms": ["never", "confined spaces", "ventilation", "flammable", "explosive", "vapours", "ignition sources"],
        "source_hint": "Dy-Mark Spray SDS Section 8",
    },
    {
        "qid": "V12",
        "category": "MSDS",
        "question": "I need to ship 400kg of Loctite 567 by road to another site. Given that it contains environmentally hazardous substances, do I need to declare this as Dangerous Goods?",
        "key_terms": ["exempt", "Special Provision AU01", "500 kg", "not", "Dangerous Goods", "road"],
        "source_hint": "Loctite 567 SDS Section 14",
    },
    {
        "qid": "V13",
        "category": "MSDS",
        "question": "I'm removing some old FBS-1 Rockwool Insulation and noticing a lot of loose fibres floating around. How should I clean up this loose material safely?",
        "key_terms": ["wet sweep", "vacuum cleaner", "PPE", "sealable plastic bag", "disposal", "dispersion"],
        "source_hint": "FBS-1 Rockwool SDS Section 6",
    },
    {
        "qid": "V14",
        "category": "MSDS",
        "question": "I'm working with Recochem Methylated Spirits and looking for the right respirator filter since the airborne levels are high. What type of filter should I select?",
        "key_terms": ["organic gases", "vapours", "boiling point", "65", "AS1716"],
        "source_hint": "Methylated Spirits SDS Section 8",
    },
    {
        "qid": "V15",
        "category": "MSDS",
        "question": "I have an empty can of Raid Flying Insect Killer and I'm about to pierce it so it crushes down flat for the recycling bin. Is this the correct disposal method?",
        "key_terms": ["do not pierce", "do not burn", "pressurized", "burst", "explode", "incineration", "recycle"],
        "source_hint": "Raid Insect Killer SDS Section 2/13",
    },

    # =========================================================================
    # Q16-Q30: Company Policies
    # =========================================================================
    {
        "qid": "V16",
        "category": "Policy",
        "question": "I am 52 years old, have worked at O'Connors for 8 years, and am currently exhausted by the after-hours on-call roster. Can I be officially excused from these shifts?",
        "key_terms": ["eligible", "7 years", "over", "50", "taken off", "On-Call Roster"],
        "source_hint": "After Hours On-Call Coverage Policy",
    },
    {
        "qid": "V17",
        "category": "Policy",
        "question": "I am rostered for on-call duties this Wednesday but a sudden personal conflict arose. I sent an email to the Service Coordinator stating I can't attend. Is my obligation fulfilled?",
        "key_terms": ["not fulfilled", "finding your own coverage", "email", "courtesy phone call", "responsible"],
        "source_hint": "After Hours On-Call Coverage Policy",
    },
    {
        "qid": "V18",
        "category": "Policy",
        "question": "I am confidentially supporting a team member who is experiencing domestic violence. They need time off for legal advice but have zero annual or sick leave left. Are there any other leave options available?",
        "key_terms": ["10 days", "paid leave", "work anniversary", "family", "domestic violence", "standalone", "separate"],
        "source_hint": "Family and Domestic Violence Policy",
    },
    {
        "qid": "V19",
        "category": "Policy",
        "question": "I am working outdoors on a roof during a severe heatwave. I have standard safety glasses and water, but the site supervisor halted my work. What specific safety and hydration protocols am I violating?",
        "key_terms": ["200ml", "15 to 20 minutes", "broad-brimmed hat", "Category 4", "UV", "buddy system", "emergency-only"],
        "source_hint": "Fatigue and Working in Heat Management Policy",
    },
    {
        "qid": "V20",
        "category": "Policy",
        "question": "I underwent a random workplace drug and alcohol test today and it returned a positive result. This is my first ever offense. Will my employment be terminated immediately?",
        "key_terms": ["not immediately", "suspended without pay", "negative test", "rehabilitation", "written warning", "second positive", "12 months", "termination"],
        "source_hint": "Fitness for Work Policy",
    },
    {
        "qid": "V21",
        "category": "Policy",
        "question": "I am operating a threading machine and am about to handle the drill chucks. Should I equip my Class 5 cut-resistant gloves to protect my hands from the sharp metal?",
        "key_terms": ["not", "entanglement", "threading machine", "rotating parts", "not mandatory", "crush", "amputation"],
        "source_hint": "Glove Policy",
    },
    {
        "qid": "V22",
        "category": "Policy",
        "question": "I am preparing to perform live electrical troubleshooting. What exact classification of gloves must I ensure are documented in my Safe Work Method Statement (SWMS) before proceeding?",
        "key_terms": ["Class 2", "Type 1", "Electrical resistant", "SWMS", "documented"],
        "source_hint": "Glove Policy",
    },
    {
        "qid": "V23",
        "category": "Policy",
        "question": "I woke up severely ill this morning. I quickly sent a text message to my manager to report my absence so I could go back to sleep. Have I complied with the leave notification rules?",
        "key_terms": ["not acceptable", "text messages", "verbally", "notice", "as soon as practicable"],
        "source_hint": "Personal/Carer's Leave Policy",
    },
    {
        "qid": "V24",
        "category": "Policy",
        "question": "I took a single sick day off on a Tuesday, immediately following a public holiday Monday. I have not used any other sick leave this year. Do I really need to provide a medical certificate?",
        "key_terms": ["yes", "two single-day", "exemption", "does not apply", "public holidays", "weekends", "medical certificate"],
        "source_hint": "Personal/Carer's Leave Policy",
    },
    {
        "qid": "V25",
        "category": "Policy",
        "question": "I am working from home connected to the company VPN. During my lunch break, I used the company laptop to download some unauthorized copyrighted movies. Is this an issue since I was on my break?",
        "key_terms": ["serious violation", "VPN", "copyrighted", "prohibited", "disciplinary", "legal action"],
        "source_hint": "Internet Usage Policy",
    },
    {
        "qid": "V26",
        "category": "Policy",
        "question": "I am stationed at an external client project site. Can I quickly step 5 metres away from the main building entrance to vape while continuing my active work duties?",
        "key_terms": ["prohibited", "external project sites", "10 metres", "designated", "lunch", "breaks"],
        "source_hint": "Smoking and Vaping Policy",
    },
    {
        "qid": "V27",
        "category": "Policy",
        "question": "My spouse was involved in an unexpected emergency today. I have zero paid personal/carer's leave accrued. Do I have any options to take time off to support them?",
        "key_terms": ["two days", "unpaid carer's leave", "immediate family", "household member", "illness", "injury", "emergency"],
        "source_hint": "Personal/Carer's Leave Policy",
    },
    {
        "qid": "V28",
        "category": "Policy",
        "question": "I overheard a supervisor making discriminatory jokes towards a subcontractor. It wasn't directed at me, so can I just ignore it to avoid workplace conflict?",
        "key_terms": ["responsibility", "standards of behaviour", "discrimination-free", "respond promptly", "sensitivity", "complaint"],
        "source_hint": "Respectful Workplace Policy",
    },
    {
        "qid": "V29",
        "category": "Policy",
        "question": "I sustained a work-related injury on site and my medical clearance only allows for restricted, light duties. Will the company accommodate this, or must I take unpaid leave until fully recovered?",
        "key_terms": ["accommodate", "suitable employment", "duties", "rehabilitation", "injured", "incapacitated"],
        "source_hint": "Rehabilitation Policy",
    },
    {
        "qid": "V30",
        "category": "Policy",
        "question": "I am managing a new fabrication process and noticed we are generating hazardous residual waste. What is the company's expectation regarding this waste and our overall environmental impact?",
        "key_terms": ["reduce", "consumption", "waste management", "safe", "responsible disposal", "environmental impact", "new processes"],
        "source_hint": "Environmental Policy",
    },

    # =========================================================================
    # Q31-Q45: Australian Standards / Technical Compliance
    # =========================================================================
    {
        "qid": "V31",
        "category": "Standards",
        "question": "I'm inspecting a cooling water system and notice that the monthly microbial testing returned a Legionella count of 15 cfu/mL. The maintenance contractor tells me they can wait until the next scheduled quarterly shutdown to disinfect. Is this acceptable?",
        "key_terms": ["not acceptable", "10 cfu/mL", "disinfection", "decontamination", "immediately"],
        "source_hint": "NSW Health Guidelines Section 5.3.1.4",
    },
    {
        "qid": "V32",
        "category": "Standards",
        "question": "I'm on a construction site and the site manager wants to use a suspended scaffolding system that hasn't been specifically tested, claiming it meets the general prefabricated scaffolding requirements. What should I advise them as an auditor?",
        "key_terms": ["insufficient", "AS 1576.4", "suspended scaffolding", "specific requirements", "AS/NZS 1576.1"],
        "source_hint": "Scaffolding Section",
    },
    {
        "qid": "V33",
        "category": "Standards",
        "question": "During a routine audit of an existing building's essential safety provisions, I noticed the hinged and pivoted fire doors are only being inspected and serviced once a year. What is the actual compliance requirement for these specific doors?",
        "key_terms": ["six-monthly", "AS 1851", "fire-resistant", "door-sets", "service schedule"],
        "source_hint": "Minister's Specification SA 76 Section 3.1(d)",
    },
    {
        "qid": "V34",
        "category": "Standards",
        "question": "I'm reviewing the architectural design for a new residential balcony balustrade utilizing safety glass, but the designer used standard empirical rules instead of limit state design methods to calculate the glass thickness. Is this allowed?",
        "key_terms": ["limit state design", "AS 1288", "Supp 1", "glass thickness", "balustrades"],
        "source_hint": "Glass Section",
    },
    {
        "qid": "V35",
        "category": "Standards",
        "question": "I'm auditing a civil engineering project's risk management plan, and the project sponsor wants to skip establishing an 'internal context,' arguing that only external environmental factors affect the project's risk. Why must I insist they include the internal context?",
        "key_terms": ["internal context", "cultures", "processes", "structures", "strategies", "strategic objectives", "credibility", "trust"],
        "source_hint": "AS/NZS IEC 62198:2015 Section 7.3.3",
    },
    {
        "qid": "V36",
        "category": "Standards",
        "question": "I am inspecting a residential timber-framed building in a basic, non-cyclonic inland area (N1 to N4 wind speeds). The builder has used span tables from AS 1684.3 to construct the frame. Is this the correct standard to apply for this specific location?",
        "key_terms": ["incorrect", "AS 1684.3", "cyclonic", "C1", "C2", "C3", "AS 1684.2", "non-cyclonic", "AS 1684.4"],
        "source_hint": "Timber Structures Section",
    },
    {
        "qid": "V37",
        "category": "Standards",
        "question": "I'm checking a newly installed commercial hot water system (used for laundry pasteurization, not an ablution system) and the water temperature at the outlet fixture is measuring exactly 55 degrees Celsius. Does this meet the required temperature for controlling microbial growth?",
        "key_terms": ["insufficient", "55", "60", "pasteurization", "minimum", "temperature losses", "radiation"],
        "source_hint": "SAA/SNZ HB32:1995 Section 3",
    },
    {
        "qid": "V38",
        "category": "Standards",
        "question": "I'm on the roof auditing a newly installed cooling tower. There are no walkways or platforms, making it impossible to safely reach the access panels for cleaning and maintenance without specialized climbing gear. Is this installation compliant?",
        "key_terms": ["non-compliant", "safe", "easy access", "AS/NZS 3666", "ladders", "ramps", "platforms", "walkways", "maintenance"],
        "source_hint": "NSW Health Guidelines Section 5.5.3",
    },
    {
        "qid": "V39",
        "category": "Standards",
        "question": "An electrician on site is installing downlights in a ceiling space heavily packed with thermal bulk insulation and asks me for the standard that details the recommended clearance distances to prevent fire hazards. Which specific standard should I direct them to?",
        "key_terms": ["AS/NZS 3000", "Wiring Rules", "clearance distances", "thermal insulating", "lighting"],
        "source_hint": "Thermal Insulating Materials Section",
    },
    {
        "qid": "V40",
        "category": "Standards",
        "question": "During my inspection of a high-rise building's mechanical ventilation system, the facility manager argues that the smoke exhaust fans only need a basic yearly operational test. What are the actual required testing frequencies for smoke exhaust systems?",
        "key_terms": ["three-monthly", "yearly", "AS 1851", "smoke exhaust", "routine tests", "record-keeping"],
        "source_hint": "Minister's Specification SA 76 Section 3.6(d)(ii)",
    },
    {
        "qid": "V41",
        "category": "Standards",
        "question": "I am reviewing an independent auditor's report for a facility's cooling tower. The report notes that the cooling tower lacks a durable sign displaying a unique identification number. Should this missing sign trigger a non-compliance report to the local government authority?",
        "key_terms": ["yes", "unique identification number", "A5", "durable", "visible", "non-compliance", "authorized officer"],
        "source_hint": "NSW Health Guidelines Section 7.4/8.5",
    },
    {
        "qid": "V42",
        "category": "Standards",
        "question": "I'm reviewing a gaseous fire-extinguishing system design layout. The piping includes a completely closed section with valves on both ends between the storage container and the discharge nozzle. Does this closed section need any specific safety protection?",
        "key_terms": ["hydrostatically tested", "pressure relief devices", "closed-section", "vent", "pressure buildup", "ISO 14520"],
        "source_hint": "ISO 14520-1:2015 Section 6.3.1.4/6.3.1.5",
    },
    {
        "qid": "V43",
        "category": "Standards",
        "question": "I am auditing a warehouse where the tenant recently changed their business model to store highly flammable liquids, but they haven't upgraded their existing portable fire extinguishers. Does the maintenance standard require any action when occupancy hazards change?",
        "key_terms": ["immediate", "reassessment", "annual check", "additional risks", "changed nature", "quantity", "fire extinguishers"],
        "source_hint": "Minister's Specification SA 76 Section 3.5(g)",
    },
    {
        "qid": "V44",
        "category": "Standards",
        "question": "I am performing a compliance check on a central emergency lighting system. The building manager claims their maintenance contractor only needs to check the power availability once a month. Are there other mandatory maintenance procedures?",
        "key_terms": ["six-monthly", "yearly", "AS/NZS 2293.2", "emergency lighting", "central systems", "common power source"],
        "source_hint": "Minister's Specification SA 76 Section 3.4(a)",
    },
    {
        "qid": "V45",
        "category": "Standards",
        "question": "A homeowner asks me if they can install a greywater diversion device without a complex treatment system to water their garden via a simple gravity feed. Where can I find the recommended diagrams and procedures for this specific untreated installation?",
        "key_terms": ["HB 326-2008", "Urban Greywater", "Handbook", "diagrams", "procedures", "untreated", "gravity"],
        "source_hint": "Greywater Systems Section",
    },
]
