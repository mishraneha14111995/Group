import numpy as np
import pandas as pd
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

n = 1000

# 1. Age: Skewed, with some outliers
age = np.random.gamma(shape=2.5, scale=10, size=n).astype(int) + 18
age[np.random.choice(n, 5)] = np.random.randint(50, 65, 5)  # Outliers

# 2. Gender
gender = np.random.choice(['Male', 'Female', 'Non-binary', 'Prefer not to say'], p=[0.38, 0.58, 0.03, 0.01], size=n)

# 3. Income: Right skewed, with some very high outliers
income_bins = ['Less than 5,000', '5,000–9,999', '10,000–19,999', '20,000–29,999', '30,000 or more']
income_probs = [0.15, 0.30, 0.32, 0.15, 0.08]
income = np.random.choice(income_bins, p=income_probs, size=n)
for i in np.random.choice(n, 8):
    income[i] = '30,000 or more'

# 4. Education
education = np.random.choice(['High School', "Bachelor’s Degree", "Master’s Degree", "Doctorate", "Other"], 
                             p=[0.21, 0.48, 0.23, 0.04, 0.04], size=n)

# 5. Nationality
nationality = np.random.choice(['Emirati', 'Expat – Arab', 'Expat – Asian', 'Expat – Western', 'Other'],
                               p=[0.12, 0.32, 0.36, 0.14, 0.06], size=n)

# 6. Purchase frequency
purchase_freq = np.random.choice(['Weekly', 'Monthly', 'Every 2–3 months', 'Rarely'], 
                                 p=[0.10, 0.44, 0.37, 0.09], size=n)

# 7. Spend (simulate via income)
spend_bins = ['Less than 100', '100–249', '250–499', '500–999', '1,000 or more']
spend_probs = {
    'Less than 5,000': [0.65, 0.22, 0.10, 0.02, 0.01],
    '5,000–9,999':     [0.42, 0.34, 0.18, 0.05, 0.01],
    '10,000–19,999':   [0.25, 0.32, 0.29, 0.10, 0.04],
    '20,000–29,999':   [0.12, 0.21, 0.35, 0.22, 0.10],
    '30,000 or more':  [0.05, 0.18, 0.27, 0.27, 0.23],
}
spend = []
for inc in income:
    spend.append(np.random.choice(spend_bins, p=spend_probs[inc]))
# Outliers
for i in np.random.choice(n, 10):
    spend[i] = '1,000 or more'

# 8. Brand preference
brand_pref = np.random.choice(['Drugstore', 'Premium', 'Korean', 'Organic/Natural', 'No preference'],
                              p=[0.24, 0.18, 0.29, 0.18, 0.11], size=n)

# 9. Purchase factor (multi-select, comma separated, 2 factors)
factors = ['Price', 'Brand reputation', 'Product ingredients', 'Online reviews', 'Social media influencers', 'Friends/Family recommendations']
purchase_factor = []
for _ in range(n):
    pf = np.random.choice(factors, size=2, replace=False)
    purchase_factor.append(", ".join(pf))

# 10. Platforms used (multi-select, association mining)
platforms = ['Instagram', 'YouTube', 'TikTok', 'Snapchat', 'Facebook']
platforms_probs = [0.85, 0.44, 0.56, 0.18, 0.09]  # Probabilities per platform
platforms_used = []
for _ in range(n):
    plat_list = [p for p, pr in zip(platforms, platforms_probs) if np.random.rand() < pr]
    if not plat_list:
        plat_list = ['None']
    platforms_used.append(", ".join(plat_list))

# 11. Trusted platform (single)
trusted_platform = []
for pu in platforms_used:
    if 'Instagram' in pu:
        trusted_platform.append(np.random.choice(['Instagram', 'YouTube', 'TikTok'], p=[0.62, 0.24, 0.14]))
    elif 'YouTube' in pu:
        trusted_platform.append(np.random.choice(['YouTube', 'TikTok'], p=[0.78, 0.22]))
    elif 'TikTok' in pu:
        trusted_platform.append('TikTok')
    else:
        trusted_platform.append('None')

# 12. Influencer count (right-skewed)
inf_cnt_bins = ['None', '1–3', '4–7', '8 or more']
inf_cnt_probs = [0.17, 0.52, 0.21, 0.10]
influencer_count = np.random.choice(inf_cnt_bins, p=inf_cnt_probs, size=n)
for i in np.random.choice(n, 7):  # Outliers: follow many
    influencer_count[i] = '8 or more'

# 13. Influencer tier followed
tier_bins = ['Micro (10k–100k)', 'Macro (100k–1M)', 'Mega (1M–10M)', 'Celebrity (10M+)', 'Not sure']
tier_probs = [0.40, 0.36, 0.14, 0.05, 0.05]
inf_tier = np.random.choice(tier_bins, p=tier_probs, size=n)

# 14. Influencer content type (multi-select, association mining)
content_types = ['Instagram Posts', 'Instagram Stories', 'Instagram Reels', 'YouTube Videos', 'TikTok Videos']
content_probs = [0.66, 0.48, 0.63, 0.32, 0.29]
content_driven = []
for _ in range(n):
    c = [ct for ct, cp in zip(content_types, content_probs) if np.random.rand() < cp]
    if not c:
        c = ['None']
    content_driven.append(", ".join(c))

# 15. Purchased after influencer (target for classification)
purchased_after_inf = []
for i in range(n):
    if 'Social media influencers' in purchase_factor[i]:
        purchased_after_inf.append(np.random.choice(['Yes', 'No'], p=[0.72, 0.28]))
    else:
        purchased_after_inf.append(np.random.choice(['Yes', 'No'], p=[0.24, 0.76]))

# 16. Likelihood to try new brand from influencer
likelihood_opts = ['Very likely', 'Likely', 'Neutral', 'Unlikely', 'Very unlikely']
likelihood_probs = [0.24, 0.32, 0.22, 0.13, 0.09]
likelihood = np.random.choice(likelihood_opts, p=likelihood_probs, size=n)

# 17. Willingness to pay extra
pay_extra_bins = ['No increase', '1–10% more', '11–25% more', '26–50% more', 'More than 50% more']
pay_extra_probs = [0.39, 0.33, 0.15, 0.09, 0.04]
pay_extra = np.random.choice(pay_extra_bins, p=pay_extra_probs, size=n)
for i in np.random.choice(n, 6):  # Outliers
    pay_extra[i] = 'More than 50% more'

# 18. Main challenge
challenges = ['Too many options', 'Not sure what works for me', 'Not enough trusted reviews', 'Products are expensive', 'Hard to find in Dubai']
challenge_probs = [0.19, 0.28, 0.19, 0.22, 0.12]
main_challenge = np.random.choice(challenges, p=challenge_probs, size=n)

# 19. Effectiveness rating
effectiveness_opts = ['Highly effective', 'Somewhat effective', 'Neutral', 'Not very effective', 'Not effective at all']
effectiveness_probs = [0.21, 0.39, 0.22, 0.12, 0.06]
effectiveness = np.random.choice(effectiveness_opts, p=effectiveness_probs, size=n)

# 20. Influential ad format
ad_format_opts = ['Standard post', 'Story', 'Reel', 'Long video (YouTube/TikTok)']
ad_format_probs = [0.19, 0.28, 0.31, 0.22]
ad_format = np.random.choice(ad_format_opts, p=ad_format_probs, size=n)

# 21. Purchase time after influencer
purchase_time_opts = ['Immediately', 'Within a week', 'Within a month', 'After a long time', 'Never']
purchase_time_probs = [0.09, 0.29, 0.36, 0.15, 0.11]
purchase_time = np.random.choice(purchase_time_opts, p=purchase_time_probs, size=n)

# 22. Part of online community
community = np.random.choice(['Yes', 'No'], p=[0.28, 0.72], size=n)

# 23. Will recommend based on influencer
recommend = []
for i in range(n):
    if purchased_after_inf[i] == 'Yes':
        recommend.append(np.random.choice(['Yes', 'No'], p=[0.83, 0.17]))
    else:
        recommend.append(np.random.choice(['Yes', 'No'], p=[0.13, 0.87]))

# 24. Turn-off in influencer marketing
turn_offs = ['Lack of authenticity', 'Over-promotion', 'Unrelatable content', 'Repetitive ads', 'None']
turn_off_probs = [0.32, 0.24, 0.16, 0.21, 0.07]
turn_off = np.random.choice(turn_offs, p=turn_off_probs, size=n)

# 25. Preferred discovery way
discovery = ['In-store', 'Online search', 'Social media/influencers', 'Friends/Family', 'Advertisements']
discovery_probs = [0.16, 0.19, 0.41, 0.15, 0.09]
preferred_discovery = np.random.choice(discovery, p=discovery_probs, size=n)

# Assemble DataFrame
df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Monthly Income (AED)': income,
    'Highest Education': education,
    'Nationality': nationality,
    'Purchase Frequency': purchase_freq,
    'Monthly Skincare Spend (AED)': spend,
    'Brand Preference': brand_pref,
    'Purchase Influencing Factors': purchase_factor,
    'Platforms Used': platforms_used,
    'Most Trusted Platform': trusted_platform,
    'Influencer Count': influencer_count,
    'Main Influencer Tier Followed': inf_tier,
    'Influencer Content Type Considered': content_driven,
    'Purchased After Influencer Recommendation': purchased_after_inf,
    'Likelihood to Try Influencer Brand': likelihood,
    'Willingness to Pay Extra': pay_extra,
    'Main Challenge': main_challenge,
    'Influencer Marketing Effectiveness': effectiveness,
    'Most Influential Ad Format': ad_format,
    'Purchase Time After Influencer': purchase_time,
    'Online Community Member': community,
    'Would Recommend Based on Influencer': recommend,
    'Biggest Influencer Marketing Turn-off': turn_off,
    'Preferred Discovery Channel': preferred_discovery,
})

# Save as CSV
csv_path = '/mnt/data/synthetic_skincare_influencer_survey.csv'
df.to_csv(csv_path, index=False)
csv_path
