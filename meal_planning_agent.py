"""
Meal Planning Agent using LangChain and Anthropic

This agent helps determine the best meals for someone who wants to lose weight
while maintaining an active lifestyle.
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def calculate_daily_calories(
    current_weight: float, target_weight: float, workouts_per_week: int
) -> str:
    """
    Calculate the recommended daily calorie intake for weight loss based on
    current weight, target weight, and activity level.

    Args:
        current_weight: Current weight in pounds
        target_weight: Target weight in pounds
        workouts_per_week: Number of workouts per week

    Returns:
        A string with the recommended daily calorie intake and macros
    """
    # Calculate BMR using Mifflin-St Jeor equation (assuming average height and age)
    # Using approximate values for a moderately active person
    bmr = current_weight * 10 + 200  # Simplified calculation

    # Activity multiplier based on workouts per week
    if workouts_per_week <= 2:
        activity_multiplier = 1.375  # Lightly active
    elif workouts_per_week <= 4:
        activity_multiplier = 1.55  # Moderately active
    else:
        activity_multiplier = 1.725  # Very active

    maintenance_calories = bmr * activity_multiplier

    # For safe weight loss (1-2 lbs per week), create a deficit
    weight_to_lose = current_weight - target_weight
    if weight_to_lose > 0:
        # Create a moderate caloric deficit (500-750 calories for 1-1.5 lbs/week loss)
        # Multiplier of 50 scales the deficit: 10 lbs to lose = 500 cal deficit
        deficit = min(750, weight_to_lose * 50)
        daily_calories = maintenance_calories - deficit
    else:
        daily_calories = maintenance_calories

    # Calculate macros (balanced approach for active individuals)
    protein_grams = current_weight * 0.8  # 0.8g per pound of body weight
    fat_grams = daily_calories * 0.25 / 9  # 25% of calories from fat
    carb_grams = (daily_calories - (protein_grams * 4) - (fat_grams * 9)) / 4

    return f"""
Calorie and Macro Recommendations:
- Maintenance Calories: {maintenance_calories:.0f} calories/day
- Target Calories for Weight Loss: {daily_calories:.0f} calories/day
- Protein: {protein_grams:.0f}g/day (for muscle preservation during weight loss)
- Carbohydrates: {carb_grams:.0f}g/day (to fuel workouts)
- Fat: {fat_grams:.0f}g/day (for hormonal balance)
- Estimated time to reach goal: {weight_to_lose / 1.5:.0f} weeks (at 1.5 lbs/week)
"""


@tool
def get_meal_suggestions(calorie_target: int, protein_target: int) -> str:
    """
    Get meal suggestions based on calorie and protein targets.

    Args:
        calorie_target: Daily calorie target
        protein_target: Daily protein target in grams

    Returns:
        A string with meal suggestions for the day
    """
    # Divide calories across meals
    breakfast_cals = int(calorie_target * 0.25)
    lunch_cals = int(calorie_target * 0.35)
    dinner_cals = int(calorie_target * 0.30)
    snack_cals = int(calorie_target * 0.10)

    return f"""
Suggested Daily Meal Plan (~{calorie_target} calories, ~{protein_target}g protein):

BREAKFAST (~{breakfast_cals} calories):
- Option 1: Greek yogurt parfait with berries and granola + 2 eggs
- Option 2: Oatmeal with protein powder, banana, and almond butter
- Option 3: Veggie omelet with whole grain toast and avocado

LUNCH (~{lunch_cals} calories):
- Option 1: Grilled chicken salad with quinoa, vegetables, and olive oil dressing
- Option 2: Turkey and avocado wrap with a side of mixed greens
- Option 3: Salmon bowl with brown rice, edamame, and vegetables

DINNER (~{dinner_cals} calories):
- Option 1: Lean beef stir-fry with vegetables and brown rice
- Option 2: Baked chicken breast with roasted sweet potato and broccoli
- Option 3: Grilled fish with quinoa and steamed vegetables

SNACKS (~{snack_cals} calories):
- Option 1: Apple with almond butter
- Option 2: Protein shake with banana
- Option 3: Mixed nuts and a piece of fruit
- Option 4: Cottage cheese with berries

TIPS FOR SUCCESS:
1. Meal prep on weekends to stay on track during busy weekdays
2. Stay hydrated - aim for 8-10 glasses of water daily
3. Eat protein with every meal to preserve muscle during weight loss
4. Time carbohydrates around workouts for optimal energy
5. Don't skip meals - consistent eating prevents overeating later
"""


@tool
def get_workout_nutrition_tips(workouts_per_week: int) -> str:
    """
    Get nutrition tips specific to workout frequency and timing.

    Args:
        workouts_per_week: Number of workouts per week

    Returns:
        Nutrition advice for active individuals
    """
    return f"""
Workout Nutrition Tips (for {workouts_per_week} workouts/week):

PRE-WORKOUT (1-2 hours before):
- Complex carbs + moderate protein
- Examples: Oatmeal with banana, whole grain toast with peanut butter
- Avoid high fat/fiber foods that slow digestion

POST-WORKOUT (within 30-60 minutes):
- Fast-digesting protein + simple carbs
- Examples: Protein shake with fruit, chicken with rice
- This is the best time for higher carb meals

ON REST DAYS:
- Slightly reduce carbohydrate intake
- Focus on protein and vegetables
- Prioritize recovery foods: lean proteins, colorful vegetables

HYDRATION:
- Drink 16-20 oz water 2-3 hours before workout
- Drink 7-10 oz every 10-20 minutes during workout
- Replenish with 16-24 oz per pound of body weight lost during exercise

SUPPLEMENTS TO CONSIDER:
- Whey or plant-based protein powder
- Creatine monohydrate (for strength training)
- Omega-3 fatty acids
- Vitamin D (especially if limited sun exposure)
"""


def create_meal_planning_agent():
    """
    Create and return a meal planning agent using LangChain and Anthropic.

    Returns:
        A configured meal planning agent
    """
    # Initialize the Anthropic model
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.7,
    )

    # Define the tools available to the agent
    tools = [calculate_daily_calories, get_meal_suggestions, get_workout_nutrition_tips]

    # System prompt for the agent
    system_prompt = """You are a helpful nutrition and meal planning expert. Your goal is to help 
people achieve their weight loss goals while maintaining energy for their active lifestyle.

You have access to the following tools:
- calculate_daily_calories: Calculate recommended calorie intake based on weight goals
- get_meal_suggestions: Get specific meal ideas based on calorie and protein targets
- get_workout_nutrition_tips: Get nutrition advice for active individuals

When helping someone:
1. First, calculate their calorie needs using the calculate_daily_calories tool
2. Then, provide meal suggestions using the get_meal_suggestions tool
3. Finally, give workout-specific nutrition tips using the get_workout_nutrition_tips tool

Be encouraging, supportive, and provide practical advice. Remember that sustainable 
weight loss is typically 1-2 pounds per week."""

    # Create the agent using langgraph's create_react_agent
    agent = create_react_agent(llm, tools, prompt=system_prompt)

    return agent


def main():
    """Main function to run the meal planning agent."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return

    # Create the agent
    agent = create_meal_planning_agent()

    # Define the user's profile
    user_profile = """
    I am currently 170 lbs and want to reach 160 lbs. I work out 3-4 times a week.
    Can you help me with:
    1. My daily calorie target for weight loss
    2. Meal suggestions that fit my goals
    3. Nutrition tips for my workout schedule
    """

    print("=" * 60)
    print("MEAL PLANNING AGENT")
    print("=" * 60)
    print(f"\nUser Profile: {user_profile.strip()}")
    print("\n" + "=" * 60)
    print("Agent Response:")
    print("=" * 60 + "\n")

    # Run the agent with the correct message format for langgraph
    result = agent.invoke({"messages": [("user", user_profile)]})

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION:")
    print("=" * 60)
    # Extract the final message from the result
    final_message = result["messages"][-1]
    print(final_message.content)


if __name__ == "__main__":
    main()
