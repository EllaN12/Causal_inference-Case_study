def plot_treatment_effect_categories(treatment_effects, thresholds=None):
    """
    Create a histogram of treatment effect categories
    
    Parameters:
    -----------
    treatment_effects: Array of treatment effects
    thresholds: List of threshold values to categorize effects
                Default is [-0.5, -0.1, 0.1, 0.5]
    
    Returns:
    --------
    None (displays the plot)
    """
    if thresholds is None:
        thresholds = [-0.5, -0.1, 0.1, 0.5]
    
    # Categorize effects
    categories = []
    for effect in treatment_effects:
        if effect <= thresholds[0]:
            categories.append('Strong Negative')
        elif effect <= thresholds[1]:
            categories.append('Moderate Negative')
        elif effect <= thresholds[2]:
            categories.append('Negligible')
        elif effect <= thresholds[3]:
            categories.append('Moderate Positive')
        else:
            categories.append('Strong Positive')
    
    # Count occurrences of each category
    category_names = ['Strong Negative', 'Moderate Negative', 'Negligible', 'Moderate Positive', 'Strong Positive']
    counts = [categories.count(cat) for cat in category_names]
    # Create a list of tuples (category, count, color)
    colors = ['#d73027', '#fc8d59', '#ffffbf', '#91cf60', '#1a9850']

    data = list(zip(categories, counts, colors))

    # Sort by count in descending order
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    sorted_categories, sorted_counts, sorted_colors = zip(*sorted_data)

    # Calculate percentages
    total = sum(counts)
    percentages = [count/total*100 for count in sorted_counts]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the bar chart with sorted data
    bars = ax.bar(sorted_categories, sorted_counts, color=sorted_colors)

    # Customize the plot
    ax.set_title('Treatment Effect Categories Distribution', fontsize=15)
    ax.set_xlabel('Effect Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

# Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,}\n({height/total*100:.1f}%)',
                ha='center', va='bottom')

# Add summary statistics
    positive_counts = counts[3] + counts[4]  # Moderate Positive + Strong Positive
    negative_counts = counts[0] + counts[1]  # Strong Negative + Moderate Negative
    neutral_counts = counts[2]  # Negligible

    positive_pct = positive_counts / total * 100
    negative_pct = negative_counts / total * 100
    neutral_pct = neutral_counts / total * 100

    plt.figtext(0.5, 0.01, 
            f'Total observations: {total:,}\n'
            f'Positive effects: {positive_counts:,} ({positive_pct:.1f}%), '
            f'Negative effects: {negative_counts:,} ({negative_pct:.1f}%), '
            f'Neutral: {neutral_counts:,} ({neutral_pct:.1f}%)', 
            ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Make layout tight
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Print the sorted counts
    print("Categories sorted by count (descending):")
    for category, count, percentage in zip(sorted_categories, sorted_counts, percentages):
        print(f"{category}: {count:,} ({percentage:.1f}%)")

    # Save the figure (optional)
    plt.savefig('treatment_effect_histogram_sorted.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

