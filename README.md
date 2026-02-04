# readme-builder
Readme builder with Ai agent 

## Meal Planning Agent

This repository includes a meal planning agent built using LangChain and Anthropic that helps determine the best meals for weight loss while maintaining an active lifestyle.

### Features

- **Calorie Calculation**: Calculates recommended daily calorie intake based on current weight, target weight, and activity level
- **Meal Suggestions**: Provides customized meal plans with breakfast, lunch, dinner, and snack options
- **Workout Nutrition Tips**: Offers nutrition advice specific to workout frequency and timing

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

2. Run the meal planning agent:

```bash
python meal_planning_agent.py
```

### Example

The agent is configured for someone who:
- Is currently 170 lbs
- Wants to reach 160 lbs
- Works out 3-4 times a week

The agent will:
1. Calculate daily calorie and macro targets for weight loss
2. Provide meal suggestions that fit those targets
3. Give nutrition tips optimized for the workout schedule

### API Reference

You can also import the agent components directly:

```python
from meal_planning_agent import (
    create_meal_planning_agent,
    calculate_daily_calories,
    get_meal_suggestions,
    get_workout_nutrition_tips
)

# Create the agent
agent = create_meal_planning_agent()

# Or use individual tools
result = calculate_daily_calories.invoke({
    'current_weight': 170,
    'target_weight': 160,
    'workouts_per_week': 4
})
print(result)
```
