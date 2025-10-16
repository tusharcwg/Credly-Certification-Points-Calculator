import os
import re
import requests
from typing import Annotated, TypedDict, Literal
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- Tool Definitions --------------------

@tool
def validate_credly_url(url: str) -> dict:
    """Validate if the provided URL is a valid Credly profile URL.
    
    Args:
        url: The Credly profile URL to validate
        
    Returns:
        dict with 'valid' (bool) and 'message' (str)
    """
    print(f"[TOOL CALL] validate_credly_url called with: url={url}")
    
    # Remove #credly or any hash fragment
    clean_url = url.split('#')[0]
    
    if clean_url.startswith("https://www.credly.com/users/"):
        result = {"valid": True, "message": "Valid Credly URL", "url": clean_url}
    else:
        result = {"valid": False, "message": "Invalid URL. Must start with 'https://www.credly.com/users/'", "url": url}
    
    print(f"[TOOL RESULT] validate_credly_url returned: {result}")
    return result


@tool
def scrape_certifications(url: str) -> dict:
    """Scrape certification data from Credly profile using public JSON API.
    
    Args:
        url: Valid Credly profile URL
        
    Returns:
        dict with certification data
    """
    print(f"[TOOL CALL] scrape_certifications called with: url={url}")
    
    certifications = []
    
    try:
        # Construct JSON API URL
        # Remove /badges if present and add .json
        base_url = url.replace('/badges', '')
        json_url = f"{base_url}/badges.json"
        
        print(f"[SCRAPING] Fetching data from: {json_url}")
        
        # Set headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
        }
        
        # Make request to Credly JSON API
        response = requests.get(json_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"[SCRAPING ERROR] HTTP {response.status_code}")
            return {
                "success": False,
                "data": {"certifications": []},
                "message": f"Failed to fetch data. HTTP Status: {response.status_code}"
            }
        
        # Parse JSON response
        data = response.json()
        print(f"[SCRAPING] Response received, parsing badges...")
        
        # Extract badges from response
        badges_data = data.get('data', [])
        
        if not badges_data:
            print("[SCRAPING] No badges found in response")
            return {
                "success": True,
                "data": {"certifications": []},
                "message": "No badges found on this profile"
            }
        
        print(f"[SCRAPING] Found {len(badges_data)} badges")
        
        # Process each badge
        for idx, badge in enumerate(badges_data):
            try:
                # Extract badge information
                title = badge.get('badge_template', {}).get('name', 'Unknown Badge')
                
                # Extract issue date
                issued_at = badge.get('issued_at', '')
                issue_date = None
                if issued_at:
                    try:
                        # Parse ISO format date
                        issue_dt = datetime.fromisoformat(issued_at.replace('Z', '+00:00'))
                        issue_date = issue_dt.strftime("%Y-%m-%d")
                    except:
                        issue_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    issue_date = datetime.now().strftime("%Y-%m-%d")
                
                # Extract expiry date
                expires_at = badge.get('expires_at')
                expiry_date = None
                if expires_at:
                    try:
                        expiry_dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                        expiry_date = expiry_dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                # Classify category based on title and badge data
                category = classify_badge_category(title, badge)
                
                cert_data = {
                    "title": title,
                    "category": category,
                    "issue_date": issue_date,
                    "expiry_date": expiry_date
                }
                
                certifications.append(cert_data)
                print(f"[SCRAPING] Badge {idx + 1}: {title} ({category}) - Expires: {expiry_date or 'Never'}")
                
            except Exception as e:
                print(f"[SCRAPING] Error processing badge {idx + 1}: {str(e)}")
                continue
        
        result = {
            "success": True,
            "data": {"certifications": certifications},
            "message": f"Successfully scraped {len(certifications)} certifications"
        }
        
    except requests.exceptions.RequestException as e:
        print(f"[SCRAPING ERROR] Request failed: {str(e)}")
        result = {
            "success": False,
            "data": {"certifications": []},
            "message": f"Network error: {str(e)}"
        }
    except Exception as e:
        print(f"[SCRAPING ERROR] {str(e)}")
        result = {
            "success": False,
            "data": {"certifications": []},
            "message": f"Error scraping profile: {str(e)}"
        }
    
    print(f"[TOOL RESULT] scrape_certifications returned: {result['message']}")
    return result


def classify_badge_category(title: str, badge_data: dict = None) -> str:
    """Classify badge category based on title keywords and badge metadata."""
    title_lower = title.lower()
    
    # Check badge metadata for level information if available
    if badge_data:
        badge_template = badge_data.get('badge_template', {})
        level = badge_template.get('level', '').lower()
        type_category = badge_template.get('type_category', '').lower()
        
        # Use Credly's own classification if available
        if 'professional' in level:
            return "Professional"
        elif 'specialty' in level or 'advanced' in level:
            return "Specialty"
        elif 'associate' in level or 'intermediate' in level:
            return "Associate"
        elif 'foundational' in level or 'beginner' in level:
            return "Foundational"
    
    # Professional level keywords
    if any(word in title_lower for word in ['professional', 'expert', 'devops engineer', 'sysops administrator', 'advanced']):
        return "Professional"
    
    # Specialty keywords
    if any(word in title_lower for word in ['specialty', 'security -', 'machine learning', 'database -', 
                                              'networking -', 'data analytics', 'sap on aws', 'advanced networking']):
        return "Specialty"
    
    # Associate keywords
    if any(word in title_lower for word in ['solutions architect - associate', 'developer - associate', 
                                              'sysops administrator - associate', 'data engineer - associate']):
        return "Associate"
    
    # Foundational keywords  
    if any(word in title_lower for word in ['cloud practitioner', 'practitioner essentials', 'essentials', 
                                              'fundamentals', 'foundations', 'certified cloud practitioner']):
        return "Foundational"
    
    # Training badge keywords
    if any(word in title_lower for word in ['training', 'partner:', 'accreditation', 'technical essentials', 
                                              'introduction to', 'getting started', 'basics', 'business professional',
                                              'sales professional', 'technical professional']):
        return "Training Badge"
    
    # Default classification based on structure
    if 'certified' in title_lower:
        if 'associate' in title_lower:
            return "Associate"
        elif 'professional' in title_lower:
            return "Professional"
        elif 'specialty' in title_lower or 'speciality' in title_lower:
            return "Specialty"
        else:
            return "Foundational"
    
    # Default to Training Badge if uncertain
    return "Training Badge"


@tool
def classify_and_validate(certifications: list) -> dict:
    """Classify certifications and validate expiry dates.
    
    Args:
        certifications: List of certification dictionaries
        
    Returns:
        dict with validated and classified certifications
    """
    print(f"[TOOL CALL] classify_and_validate called with {len(certifications)} certifications")
    
    current_date = datetime.now()
    validated_certs = []
    
    for cert in certifications:
        cert_copy = cert.copy()
        
        # Check expiry
        if cert['expiry_date']:
            try:
                expiry_date = datetime.strptime(cert['expiry_date'], "%Y-%m-%d")
                cert_copy['is_valid'] = expiry_date > current_date
                cert_copy['status'] = "Valid" if cert_copy['is_valid'] else "Expired"
                
                # Calculate days until expiry
                days_diff = (expiry_date - current_date).days
                if cert_copy['is_valid'] and days_diff < 90:
                    cert_copy['status'] = f"Valid (Expires in {days_diff} days)"
                    
            except Exception as e:
                print(f"[VALIDATION] Error parsing expiry date: {e}")
                cert_copy['is_valid'] = True
                cert_copy['status'] = "Valid (Unknown Expiry)"
        else:
            cert_copy['is_valid'] = True
            cert_copy['status'] = "Valid (No Expiry)"
        
        validated_certs.append(cert_copy)
    
    result = {
        "validated_certifications": validated_certs,
        "total_certs": len(validated_certs),
        "valid_certs": sum(1 for c in validated_certs if c['is_valid']),
        "expired_certs": sum(1 for c in validated_certs if not c['is_valid'])
    }
    
    print(f"[TOOL RESULT] classify_and_validate returned: {result['valid_certs']} valid, {result['expired_certs']} expired")
    return result


@tool
def calculate_points(validated_certifications: list) -> dict:
    """Calculate total points based on valid certifications.
    
    Args:
        validated_certifications: List of validated certification dictionaries
        
    Returns:
        dict with points breakdown and total
    """
    print(f"[TOOL CALL] calculate_points called")
    
    # Points mapping
    points_map = {
        "Training Badge": 10,
        "Foundational": 10,
        "Associate": 15,
        "Professional": 20,
        "Specialty": 20
    }
    
    total_points = 0
    points_breakdown = []
    category_summary = {}
    
    for cert in validated_certifications:
        if cert['is_valid']:
            category = cert['category']
            points = points_map.get(category, 0)
            total_points += points
            
            points_breakdown.append({
                "title": cert['title'],
                "category": category,
                "points": points,
                "status": cert['status']
            })
            
            # Count by category
            if category in category_summary:
                category_summary[category]['count'] += 1
                category_summary[category]['points'] += points
            else:
                category_summary[category] = {'count': 1, 'points': points}
    
    result = {
        "total_points": total_points,
        "points_breakdown": points_breakdown,
        "category_summary": category_summary,
        "valid_count": len(points_breakdown)
    }
    
    print(f"[TOOL RESULT] calculate_points returned: {total_points} total points from {len(points_breakdown)} valid certs")
    return result


# List of available tools
tools = [validate_credly_url, scrape_certifications, classify_and_validate, calculate_points]

# -------------------- LLM Setup --------------------

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)

# -------------------- State Definition --------------------

class CredlyState(TypedDict):
    messages: Annotated[list, add_messages]
    url: str
    stage: str
    validation_result: dict
    certifications: list
    validated_certs: dict
    points_result: dict
    error: str

# -------------------- Graph Nodes --------------------

def agent_node(state: CredlyState):
    """Agent decides what action to take based on current state."""
    print("\n=== AGENT NODE ===")
    
    # Create system message to guide the agent
    system_msg = SystemMessage(content="""You are a Credly Certification Points Calculator assistant.
Your job is to help users calculate their certification points by:
1. Validating their Credly URL
2. Scraping certification data from Credly's public API
3. Validating certifications and checking expiry dates
4. Calculating points based on certification categories

Follow the workflow step by step. Use the tools in this order:
1. validate_credly_url - to check if URL is valid
2. scrape_certifications - to get certification data from Credly (quick JSON API)
3. classify_and_validate - to validate certifications and check expiry
4. calculate_points - to compute total points

After all tools are executed, provide a comprehensive summary showing:
- Total certifications found
- Breakdown by category (Training Badge, Foundational, Associate, Professional, Specialty)
- Valid vs expired certifications
- Points earned per category
- Total points

Format the final output in a clear, organized manner with certification details listed.""")
    
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def should_continue(state: CredlyState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return "end"


# -------------------- Build Graph --------------------

graph_builder = StateGraph(CredlyState)

# Add nodes
graph_builder.add_node("agent", agent_node)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
graph_builder.add_edge("tools", "agent")

# Compile graph
graph = graph_builder.compile()

# -------------------- Main Application --------------------

def run_credly_calculator():
    """Main function to run the Credly Certification Points Calculator."""
    print("=" * 70)
    print("CREDLY CERTIFICATION POINTS CALCULATOR")
    print("=" * 70)
    print("\nThis application calculates certification points from your Credly profile.")
    print("Using Credly's public JSON API for fast and accurate data retrieval.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Please enter your Credly profile URL: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting application. Goodbye!")
            break
        
        if not user_input:
            print("Error: Please provide a URL.\n")
            continue
        
        print("\n" + "=" * 70)
        print("PROCESSING YOUR REQUEST...")
        print("=" * 70)
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=f"Calculate certification points for this Credly profile: {user_input}")],
            "url": user_input,
            "stage": "input",
            "validation_result": {},
            "certifications": [],
            "validated_certs": {},
            "points_result": {},
            "error": ""
        }
        
        try:
            # Run the graph
            result = graph.invoke(initial_state)
            
            # Display final result
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            
            final_message = result["messages"][-1]
            print(f"\n{final_message.content}\n")
            
        except Exception as e:
            print(f"\nError processing request: {str(e)}\n")
            import traceback
            traceback.print_exc()
        
        print("=" * 70)
        print()


if __name__ == "__main__":
    run_credly_calculator()
