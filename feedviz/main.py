import os
from dotenv import load_dotenv
load_dotenv("feedviz/.env")
from crewai import  Crew, Process
from feedviz.agents.data_processing_agent import data_processing_agent, data_processing_task
from feedviz.config.settings import settings


def build_crew() -> Crew:
    agents=[
        data_processing_agent,
    ]
    tasks=[
        data_processing_task
    ]
    crew=Crew(agents=agents,tasks=tasks,process=Process.sequential,verbose=True)
    return crew

def run_pipeline(csv_path: str=None)->dict:
    crew = build_crew()
    inputs={
        "csv_path": str(csv_path or settings.feedback_csv),
    }
    result=crew.kickoff(inputs=inputs)
    return {
        "status":"success",
        "result": result,
    }
if __name__=="__main__":
    output=run_pipeline()
    print("\n[PIPELINE COMPLETE]")
    print(output)