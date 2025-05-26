import pandas as pd
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import time
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Filter out httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class PhysicsAgentResult(BaseModel):
    reasoning: str = Field(description="The reasoning of the agent.")
    selected_features: List[Literal["IUPAC name", "Formula", "Molecular weight", "XLogP",
        "H-bond donors", "H-bond acceptors", "Rotatable bonds",
        "Polar Surface Area", "Synonyms"]] = Field(
            description="The features that the agent has selected for the molecule.",
            min_items=3,
            max_items=5
        )
    feature_weights: List[float] = Field(
        description="The weights of the features that the agent has selected for the molecule. Their sum must be equal to 1.",
        min_items=3,
        max_items=5
    )

class PhysicsCriticAgentResult(BaseModel):
    reasoning: str = Field(description="The reasoning of the critic agent.")
    is_valid: bool = Field(description="Whether the agent's result is valid.")

@dataclass
class CritiqueHistory:
    iteration: int
    critique: str
    is_valid: bool

class PhysicsAgent:
    def __init__(self, model_name: str = "llama3"):
        self.llm = ChatOllama(model=model_name).with_structured_output(PhysicsAgentResult)
        self.system_message = SystemMessage(content="""You are a physics agent that analyzes molecules and selects relevant features for a target property.
        You must select between 3-5 features from the available options and assign weights to them.
        The weights must sum to 1.0.
        Available features are: IUPAC name, Formula, Molecular weight, XLogP, H-bond donors, 
        H-bond acceptors, Rotatable bonds, Polar Surface Area, and Synonyms.
        Provide your response in a structured format with clear reasoning.""")

    def analyze(self, molecule_description: str, target_property_information: str, critique_input: str = "") -> PhysicsAgentResult:
        messages = [
            self.system_message,
            HumanMessage(content=f"Analyze this molecule InChi String and select relevant features: {molecule_description}"
                         f"and the target property information: {target_property_information}"
                         f"and the critique input: {critique_input}"),
        ]
        
        response = self.llm.invoke(messages)
        logger.info(f"Physics Agent Response: {response}")
        return response

class PhysicsCritic:
    def __init__(self, model_name: str = "llama3"):
        self.llm = ChatOllama(model=model_name).with_structured_output(PhysicsCriticAgentResult)
        self.system_message = SystemMessage(content="""You are a materials scientist that validates the output of the junior materials scientist.
        Your role is to check if the selected features and their weights are appropriate for the target property.
        Consider the following criteria:
        1. Are the selected features relevant to the target property?
        2. Do the weights reflect the relative importance of each feature?
        3. Are there any critical features missing?
        4. Are the weights properly distributed (sum to 1.0)?
        
        Provide your response with clear reasoning and a boolean indicating if the selection is valid.""")

    def validate(self, physics_result: PhysicsAgentResult, target_property_information: str) -> PhysicsCriticAgentResult:
        messages = [
            self.system_message,
            HumanMessage(content=f"Validate this physics agent result for the target property '{target_property_information}':\n"
                         f"Selected features: {physics_result.selected_features}\n"
                         f"Feature weights: {physics_result.feature_weights}\n"
                         f"Reasoning: {physics_result.reasoning}"),
        ]
        
        response = self.llm.invoke(messages)
        logger.info(f"Physics Critic Response: {response}")
        return response

class MoleculeAnalyzer:
    def __init__(self, physics_agent: PhysicsAgent, physics_critic: PhysicsCritic):
        self.physics_agent = physics_agent
        self.physics_critic = physics_critic

    def analyze_molecule(self, molecule_description: str, target_property_information: str, max_iterations: int = 3) -> PhysicsAgentResult:
        current_result = self.physics_agent.analyze(molecule_description, target_property_information)
        
        for _ in range(max_iterations):
            critique = self.physics_critic.validate(current_result, target_property_information)
            
            if critique.is_valid:
                return current_result
                
            current_result = self.physics_agent.analyze(
                molecule_description,
                target_property_information,
                critique.reasoning
            )
        
        return current_result

class QM9DatasetProcessor:
    def __init__(self, analyzer: MoleculeAnalyzer):
        self.analyzer = analyzer
        self.columns_to_keep = [
            'filename',
            'index',
            'agent_reasoning',
            'selected_features',
            'feature_weights',
            'critique_history',
            'final_critique',
            'iterations_needed'
        ]

    def process_dataset(self, csv_path: str, target_property: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if sample_size:
            df = df[:sample_size]

        df['agent_reasoning'] = None
        df['selected_features'] = None
        df['feature_weights'] = None
        df['critique_history'] = None
        df['iterations_needed'] = None
        df['final_critique'] = None

        for idx, row in df.iterrows():
            logger.info(f"Processing molecule {idx + 1}/{len(df)}")
            
            try:
                result = self._process_single_molecule(row['InChI  (GDB-9)'], target_property)
                self._update_dataframe(df, idx, result)
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error processing molecule {idx + 1}: {str(e)}")
                continue

        return df[self.columns_to_keep]

    def _process_single_molecule(self, inchi: str, target_property: str) -> Dict[str, Any]:
        critique_history = []
        iterations = 0
        valid = False

        current_result = self.analyzer.physics_agent.analyze(inchi, target_property)
        
        while iterations < 3:
            iterations += 1
            critique = self.analyzer.physics_critic.validate(current_result, target_property)
            critique_history.append(CritiqueHistory(
                iteration=iterations,
                critique=critique.reasoning,
                is_valid=critique.is_valid
            ))
            
            if critique.is_valid:
                valid = True
                break
                
            current_result = self.analyzer.physics_agent.analyze(
                inchi,
                target_property,
                critique.reasoning
            )

        # If never valid after 3 tries, count that as 4th "failed" iteration,
        if not valid:
            iterations += 1

        return {
            'agent_reasoning': current_result.reasoning,
            'selected_features': ','.join(current_result.selected_features),
            'feature_weights': ','.join(map(str, current_result.feature_weights)),
            'critique_history': str(critique_history),
            'iterations_needed': iterations,
            'final_critique': critique_history[-1].critique if critique_history else None
        }

    def _update_dataframe(self, df: pd.DataFrame, idx: int, result: Dict[str, Any]) -> None:
        for key, value in result.items():
            df.at[idx, key] = value

class QM9AnalysisPipeline:
    QM9_PROPERTIES = [
        "Dipole Moment",
        "Isotropic polarizability",
        "HOMO",
        "LUMO",
        "HOMO-LUMO gap",
        "Electronic spatial extent",
        "Zero point vibrational energy",
        "Internal energy at 0K",
        "Internal energy at 298.15K",
    ]

    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        physics_agent = PhysicsAgent()
        physics_critic = PhysicsCritic()
        analyzer = MoleculeAnalyzer(physics_agent, physics_critic)
        self.processor = QM9DatasetProcessor(analyzer)

    def run_analysis(self, sample_size: Optional[int] = None):
        for target_property in self.QM9_PROPERTIES:
            logger.info(f"\nProcessing property: {target_property}")
            
            output_file = self.output_dir / f"results_{target_property.replace(' ', '_').replace('-', '_')}.csv"
            
            results_df = self.processor.process_dataset(
                self.input_file,
                target_property,
                sample_size
            )
            
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            time.sleep(2)  # Rate limiting between properties

def main():
    input_file = "./data/qm9_parsed_merged_data.csv"
    output_dir = "./agent_results"
    
    pipeline = QM9AnalysisPipeline(input_file, output_dir)
    pipeline.run_analysis()  # Process first 5 molecules for testing if sample_size=5

if __name__ == "__main__":
    main()
