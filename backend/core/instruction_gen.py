import re
import json
from backend.core.dataset_loader import LegalDataset

class LegalInstructionGenerator:
    """
    Generates high-quality SFT (Supervised Fine-Tuning) data from legal text.
    Focuses on GROUNDING and REFUSAL behavior.
    """
    def __init__(self):
        self.templates = [
            # Definitions & Provisions
            {"q": "What is {section}?", "a": "According to the provided context, {section} refers to {content}."},
            {"q": "Explain the provision of {section}.", "a": "Based on the retrieved legal text, {section} states that {content}."},
            {"q": "What does {section} of the law define?", "a": "As per the legal documents, {section} defines {content}."},
            
            # Punishments & Consequences
            {"q": "What is the punishment under {section}?", "a": "According to the provided context, the punishment or consequence under {section} is related to {content}. (Consult the specific Act for precise sentencing)."},
            {"q": "What are the penalties associated with {section}?", "a": "The legal text indicates that {section} involves penalties or proceedings concerning {content}."},
            
            # Procedural & Evidence
            {"q": "How is {section} applied in court?", "a": "In accordance with procedural rules, {section} is applied to matters involving {content}."},
            {"q": "What evidence is required under {section}?", "a": "The context suggests that {section} relates to the evidentiary requirements for {content}."},
            
            # Scenarios
            {"q": "In a case of {title}, which section applies?", "a": "In cases involving {title}, {section} is often relevant as it deals with {content}."},
            {"q": "If someone is accused of {title}, what is the relevant law?", "a": "Based on the provided legal context, the relevant law for {title} would be {section}, which addresses {content}."},
        ]

    def generate_dataset(self, text, output_path="backend/core/legal_sft_data.json"):
        print("DEBUG: Analyzing text for Section patterns...")
        # Regex to find Sections (e.g., Section 302, Section 420)
        # We look for "Section [Number]" followed by a title and content
        pattern = r"(Section\s+\d+[A-Z]?)\s+(.*?)\.\s+(.*?)(?=Section\s+\d+|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        
        sft_data = []
        
        # 1. Positive Samples (Grounded)
        for i, (section, title, content) in enumerate(matches):
            if len(content.strip()) < 50: continue
            
            clean_content = content.strip().replace("\n", " ")
            
            # Use templates to generate 2-3 variations per section
            for template in self.templates:
                sft_data.append({
                    "instruction": template["q"].format(section=section),
                    "context": f"{section} {title}. {clean_content}",
                    "answer": template["a"].format(section=section, content=clean_content[:300])
                })
                
            # 2. Negative Samples (Refusal)
            if i % 10 == 0:
                sft_data.append({
                    "instruction": f"What is Section {999+i}?",
                    "context": f"{section} {title}. {clean_content}",
                    "answer": f"The provided context does not contain information about Section {999+i}. It only discusses {section}."
                })

        print(f"DEBUG: Generated {len(sft_data)} instruction-tuning samples.")
        
        with open(output_path, "w") as f:
            json.dump(sft_data, f, indent=2)
        
        return sft_data

if __name__ == "__main__":
    # Test generation
    loader = LegalDataset()
    generator = LegalInstructionGenerator()
    generator.generate_dataset(loader.raw_text)
