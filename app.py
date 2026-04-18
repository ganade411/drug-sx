import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Groq client ──────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"   # Current Groq model

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"status": "online", "message": "DrugsX AI Backend is running", "model": MODEL})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.get_json(force=True)
        protein = (data.get("protein") or "").strip()
        drug    = (data.get("drug")    or "").strip()

        if not protein or not drug:
            return jsonify({"error": "Both 'protein' and 'drug' fields are required."}), 400

        prompt = f"""You are a computational drug-discovery expert with deep knowledge of molecular biology, pharmacokinetics, and protein-ligand docking.

Analyze the following protein-drug interaction and return ONLY a valid JSON object — no markdown, no explanation, nothing else.

Protein Sequence : {protein[:500]}
Drug SMILES      : {drug[:300]}

Return this exact JSON schema (all fields required):
{{
  "binding_score"      : <float 0.0–1.0, higher = stronger binding>,
  "confidence"         : <float 0.0–1.0, model confidence in this prediction>,
  "interaction_type"   : "<one of: hydrophobic | hydrogen_bond | ionic | van_der_waals | mixed | unknown>",
  "mechanism"          : "<1-3 sentences describing the dominant binding mechanism>",
  "drug_likeness"      : "<Lipinski / drug-likeness assessment in 1 sentence>",
  "side_effects_risk"  : "<low | moderate | high>",
  "bioavailability"    : "<poor | moderate | good>",
  "selectivity"        : "<low | moderate | high>",
  "recommendation"     : "<1 sentence: proceed / further investigation needed / unlikely candidate>"
}}

Scoring guide:
  0.85–1.00 → Exceptional binding (clinical candidate)
  0.65–0.84 → Strong binding (promising lead)
  0.45–0.64 → Moderate binding (needs optimisation)
  0.25–0.44 → Weak binding (poor candidate)
  0.00–0.24 → No significant interaction"""

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a pharmaceutical AI. Always respond with a single, valid JSON object only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=512,
        )

        raw = completion.choices[0].message.content.strip()
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            raise ValueError("No JSON found in model response")

        analysis = json.loads(json_match.group())
        analysis["binding_score"] = max(0.0, min(1.0, float(analysis.get("binding_score", 0.5))))
        analysis["confidence"]    = max(0.0, min(1.0, float(analysis.get("confidence",    0.7))))

        return jsonify({
            "binding_score"    : analysis["binding_score"],
            "confidence"       : analysis["confidence"],
            "interaction_type" : analysis.get("interaction_type",  "unknown"),
            "mechanism"        : analysis.get("mechanism",         "Not available"),
            "drug_likeness"    : analysis.get("drug_likeness",     "Not assessed"),
            "side_effects_risk": analysis.get("side_effects_risk", "moderate"),
            "bioavailability"  : analysis.get("bioavailability",   "moderate"),
            "selectivity"      : analysis.get("selectivity",       "moderate"),
            "recommendation"   : analysis.get("recommendation",    "Further investigation needed"),
            "protein_preview"  : protein[:60] + ("…" if len(protein) > 60 else ""),
            "drug_preview"     : drug[:60]    + ("…" if len(drug)    > 60 else ""),
            "model_used"       : MODEL,
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse model response: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """AI chatbot endpoint — supports multi-turn conversation history."""
    try:
        data = request.get_json(force=True)
        user_message = (data.get("message") or "").strip()
        history      = data.get("history", [])

        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DrugsX AI, an expert bioinformatics and drug-discovery assistant. "
                    "You help researchers understand protein sequences, SMILES notation, drug-likeness, "
                    "pharmacokinetics, binding affinities, and the drug development pipeline. "
                    "Be concise, scientifically accurate, and use markdown formatting for clarity. "
                    "Use bullet points, bold text, and code blocks where appropriate."
                )
            }
        ]

        for turn in history[-10:]:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=1024,
        )

        reply = completion.choices[0].message.content.strip()
        return jsonify({"reply": reply, "model": MODEL})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-smiles", methods=["POST"])
def analyze_smiles():
    """Quick SMILES-only analysis — drug-likeness and properties."""
    try:
        data  = request.get_json(force=True)
        smiles = (data.get("smiles") or "").strip()

        if not smiles:
            return jsonify({"error": "SMILES string is required."}), 400

        prompt = f"""Analyze this SMILES string and return ONLY a valid JSON object.

SMILES: {smiles[:300]}

Return exactly:
{{
  "compound_name"    : "<best guess at compound class or name>",
  "molecular_formula": "<chemical formula if deducible>",
  "drug_likeness"    : "<Lipinski Rule of Five assessment in 1-2 sentences>",
  "toxicity_alert"   : "<none | low | moderate | high>",
  "functional_groups": ["<list up to 5 key functional groups>"],
  "therapeutic_area" : "<most likely therapeutic area>",
  "notes"            : "<any notable structural features in 1-2 sentences>"
}}"""

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a medicinal chemistry AI. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512,
        )

        raw = completion.choices[0].message.content.strip()
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            raise ValueError("No JSON in model response")

        result = json.loads(json_match.group())
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse model response: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
#  NEW: PHARMACOKINETICS ENDPOINT
# ══════════════════════════════════════════════════════════════════
@app.route("/pharmacokinetics", methods=["POST"])
def pharmacokinetics():
    """Full ADMET + PK profile for a given drug SMILES."""
    try:
        data   = request.get_json(force=True)
        smiles = (data.get("smiles") or "").strip()
        name   = (data.get("drug_name") or "Unknown Compound").strip()

        if not smiles:
            return jsonify({"error": "SMILES string is required."}), 400

        prompt = f"""You are a senior pharmacokinetics scientist and ADMET modelling expert.

Analyze the following drug compound and return ONLY a valid JSON object with no markdown or explanation.

Drug Name  : {name}
SMILES     : {smiles[:300]}

Return this EXACT JSON schema. All numeric fields must be floats 0.0–1.0 (probability/normalised score) unless a unit is specified:
{{
  "compound_name"        : "<confirmed or best-guess compound name>",
  "molecular_formula"    : "<chemical formula>",
  "molecular_weight"     : <number in Daltons>,
  "logP"                 : <number, octanol-water partition coefficient, typical range -2 to 6>,
  "hbd"                  : <integer, hydrogen bond donors>,
  "hba"                  : <integer, hydrogen bond acceptors>,
  "tpsa"                 : <number in Å², topological polar surface area>,
  "rotatable_bonds"      : <integer>,
  "lipinski_compliant"   : <true | false>,

  "absorption"           : {{
    "score"              : <float 0–1, overall absorption probability>,
    "oral_bioavailability": "<poor | moderate | good>",
    "pgp_substrate"      : "<yes | no | likely>",
    "caco2_permeability" : "<low | medium | high>",
    "hia"                : "<low | medium | high (human intestinal absorption)>",
    "notes"              : "<1-2 sentences>"
  }},

  "distribution"         : {{
    "score"              : <float 0–1>,
    "vd"                 : "<low | moderate | high (volume of distribution)>",
    "bbb_penetration"    : "<yes | no | unlikely (blood-brain barrier)>",
    "plasma_protein_binding": "<low | moderate | high>",
    "notes"              : "<1-2 sentences>"
  }},

  "metabolism"           : {{
    "score"              : <float 0–1, favourable metabolism score>,
    "cyp1a2_inhibitor"   : "<yes | no>",
    "cyp2c19_inhibitor"  : "<yes | no>",
    "cyp2c9_inhibitor"   : "<yes | no>",
    "cyp2d6_inhibitor"   : "<yes | no>",
    "cyp3a4_inhibitor"   : "<yes | no>",
    "primary_pathway"    : "<CYP3A4 | CYP2D6 | CYP2C9 | hydrolysis | other>",
    "half_life"          : "<short <2h | medium 2-12h | long >12h>",
    "notes"              : "<1-2 sentences>"
  }},

  "excretion"            : {{
    "score"              : <float 0–1, favourable excretion score>,
    "primary_route"      : "<renal | hepatic | biliary | mixed>",
    "renal_clearance"    : "<low | moderate | high>",
    "notes"              : "<1-2 sentences>"
  }},

  "toxicity"             : {{
    "score"              : <float 0–1, safety score — higher is safer>,
    "ames_test"          : "<positive | negative (mutagenicity)>",
    "herg_inhibition"    : "<yes | no | unlikely>",
    "hepatotoxicity"     : "<low | moderate | high>",
    "carcinogenicity"    : "<low | moderate | high>",
    "ld50_estimate"      : "<low | moderate | high (estimated acute toxicity)>",
    "notes"              : "<1-2 sentences>"
  }},

  "overall_pk_score"     : <float 0–1, aggregated PK/ADMET favourability>,
  "drug_class"           : "<best therapeutic class estimate>",
  "development_stage"    : "<preclinical | phase_i | phase_ii | phase_iii | approved (estimated based on profile)>",
  "key_concerns"         : ["<up to 3 key PK/safety concerns>"],
  "recommendations"      : "<2-3 sentences on next steps for optimisation or development>"
}}"""

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a pharmacokinetics AI. Respond only with a single valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=1200,
        )

        raw = completion.choices[0].message.content.strip()
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            raise ValueError("No JSON in model response")

        result = json.loads(json_match.group())

        # Clamp all top-level scores
        for key in ["overall_pk_score"]:
            if key in result:
                result[key] = max(0.0, min(1.0, float(result[key])))
        for section in ["absorption", "distribution", "metabolism", "excretion", "toxicity"]:
            if section in result and "score" in result[section]:
                result[section]["score"] = max(0.0, min(1.0, float(result[section]["score"])))

        result["model_used"] = MODEL
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse model response: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
#  NEW: ADVANCED PROTEIN-DRUG INTERACTION ENDPOINT
# ══════════════════════════════════════════════════════════════════
@app.route("/protein-drug-interaction", methods=["POST"])
def protein_drug_interaction():
    """Deep protein-drug interaction analysis with binding site residues and network."""
    try:
        data    = request.get_json(force=True)
        protein = (data.get("protein") or "").strip()
        drug    = (data.get("drug")    or "").strip()
        protein_name = (data.get("protein_name") or "Unknown Protein").strip()
        drug_name    = (data.get("drug_name")    or "Unknown Drug").strip()

        if not protein or not drug:
            return jsonify({"error": "Both 'protein' and 'drug' fields are required."}), 400

        prompt = f"""You are a structural bioinformatics and molecular docking expert with deep knowledge of protein-ligand interactions, binding site analysis, and drug resistance mechanisms.

Perform a comprehensive protein-drug interaction analysis and return ONLY a valid JSON object.

Protein Name    : {protein_name}
Protein Sequence: {protein[:400]}
Drug Name       : {drug_name}
Drug SMILES     : {drug[:200]}

Return this EXACT JSON schema:
{{
  "binding_affinity"     : <float 0–1, predicted binding affinity>,
  "docking_score"        : <float, estimated docking score in kcal/mol, typically -12 to -3>,
  "confidence"           : <float 0–1>,

  "binding_site"         : {{
    "pocket_volume"      : "<small <200Å³ | medium 200-600Å³ | large >600Å³>",
    "druggability"       : "<low | moderate | high>",
    "active_site"        : "<yes | no (is drug binding at the active site?)>",
    "allosteric"         : "<yes | no>"
  }},

  "key_residues"         : [
    {{
      "position"         : <estimated residue number>,
      "amino_acid"       : "<3-letter code e.g. ARG, LYS, ASP>",
      "one_letter"       : "<single letter code>",
      "interaction_type" : "<H-bond | hydrophobic | ionic | pi-stacking | van_der_waals>",
      "contribution"     : "<critical | major | minor>",
      "distance_angstrom": <estimated distance in Angstroms 1.5-5.0>
    }}
  ],

  "interaction_network"  : {{
    "hydrogen_bonds"     : <integer count>,
    "hydrophobic_contacts": <integer count>,
    "ionic_interactions" : <integer count>,
    "pi_stacking"        : <integer count>,
    "van_der_waals"      : <integer count>,
    "total_contacts"     : <integer total>
  }},

  "protein_analysis"     : {{
    "secondary_structure" : "<alpha-helix dominated | beta-sheet dominated | mixed | disordered>",
    "flexibility"         : "<rigid | semi-flexible | flexible>",
    "functional_class"    : "<enzyme | receptor | transporter | channel | structural | other>",
    "target_family"       : "<kinase | GPCR | protease | nuclear receptor | ion channel | other>"
  }},

  "drug_analysis"        : {{
    "binding_mode"        : "<orthosteric | allosteric | covalent | competitive | non-competitive>",
    "pharmacophore_match" : "<excellent | good | moderate | poor>",
    "strain_energy"       : "<low | moderate | high>",
    "induced_fit"         : "<yes | no (does protein require conformational change?)>"
  }},

  "selectivity_profile"  : {{
    "target_selectivity"  : "<low | moderate | high>",
    "off_target_risk"     : "<low | moderate | high>",
    "resistance_risk"     : "<low | moderate | high>",
    "notes"               : "<1-2 sentences on selectivity>"
  }},

  "thermodynamics"       : {{
    "delta_g"             : <estimated free energy of binding in kcal/mol, typically -15 to -3>,
    "enthalpy_driven"     : "<yes | no>",
    "entropy_driven"      : "<yes | no>",
    "ki_estimate"         : "<nanomolar | micromolar | millimolar (estimated inhibition constant)>"
  }},

  "clinical_relevance"   : {{
    "therapeutic_potential": "<low | moderate | high>",
    "novelty"             : "<me-too | improved | novel>",
    "development_readiness": "<early-stage | lead-optimization | preclinical | clinical>",
    "notes"               : "<2-3 sentences on clinical potential>"
  }},

  "visualization_data"   : {{
    "interaction_strengths": {{
      "h_bond_strength"    : <float 0–1>,
      "hydrophobic_strength": <float 0–1>,
      "ionic_strength"     : <float 0–1>,
      "vdw_strength"       : <float 0–1>,
      "pi_strength"        : <float 0–1>
    }}
  }},

  "recommendations"      : "<2-3 sentences on how to improve this interaction>"
}}

Provide exactly 5-8 key_residues that are most important for the interaction."""

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a structural bioinformatics AI. Respond only with a single valid JSON object, no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=1800,
        )

        raw = completion.choices[0].message.content.strip()
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            raise ValueError("No JSON in model response")

        result = json.loads(json_match.group())

        # Clamp scores
        for key in ["binding_affinity", "confidence"]:
            if key in result:
                result[key] = max(0.0, min(1.0, float(result[key])))

        result["protein_preview"] = protein[:60] + ("…" if len(protein) > 60 else "")
        result["drug_preview"]    = drug[:60]    + ("…" if len(drug)    > 60 else "")
        result["model_used"]      = MODEL
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse model response: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
