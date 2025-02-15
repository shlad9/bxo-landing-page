import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

m = pd.read_excel("Movies _ TV Show Database.xlsx")
w = pd.get_dummies(m, columns=['Primary Genre', 'Secondary Genre', 'Mood & Tone', 'Main Cast', 'Production Company', 'Revenue'])
k = pd.read_excel("responses_movies_w0.xlsx")
o = pd.get_dummies(k, columns=['Which of these best describe you?', 'How would you best describe yourself?','Which of these best describe you?.1', 'What time of day do you usually watch content?', 'What is your favorite genre?'])

userdf = o
moviedf = w
movies_df = w
users_df = o

users_df  = users_df .rename(columns={'Thank you for taking part in the Movie Survey. \n\nPlease enter your full name': 'user_id'})
users_df = users_df.rename(columns= {'Rate *"Dark Knight - Batman Begins"*': 'The Dark Knight - Batman Begins',
                                     'Rate *"Harry Potter and the Sorcerer\'s Stone"*' : 'Harry Potter and the Sorcerer\'s Stone',
                                     'Rate *"Toy Story"*' : 'Toy Story',
                                     'Rate *"Mean Girls"*' : 'Mean Girls',
                                     'Rate *"The Social Network"*' : 'The Social Network',
                                     'Rate *"Get Out"*' : 'Get Out',
                                     'Rate *"Spider-Man Into the Spider-verse"*' : 'Spider-man Into the Spider-verse',
                                     'Rate *"Spirited Away"*' : 'Spirited Away',
                                     'Rate *"Insidious"*' : 'Insidious',
                                     'Rate *"Inception"*' : 'Inception',
                                     'Rate *"The Notebook"*' : 'The Notebook'
                                    })

# non watched movies should get removed from the base list and added to the longer list
# need to find a way to user better string assignments to go from the survey to the columns (instead of Rate XYZ, just XYZ to pair
# how do we think about 0 weights given to unwatched movies? Is not watching something a factor of hating so much u wont, or
# not having heard of it?

ratings = users_df[['user_id', 'The Dark Knight - Batman Begins', 'Harry Potter and the Sorcerer\'s Stone',
                   'Toy Story', 'Mean Girls', 'The Social Network', 'Get Out',
                    'Spider-man Into the Spider-verse', 'Spirited Away', 'Insidious', 
                    'Inception', 'The Notebook'
                   ]]


l = ratings
l = l.fillna(0)
l = l.set_index('user_id')
l = l.transpose()
user_ratings = l.to_dict()

users_df = users_df[['user_id', 'How likely are you to Binge a show?', 'How old are you?',
                     'Which of these best describe you?_The Explorer – I love discovering new things and trying out what’s fresh and different.',
 'How would you best describe yourself?_Ambivert',
 'How would you best describe yourself?_Introvert',
 'Which of these best describe you?.1_The Loyal Fan – I stick to my favorite genres, franchises, and creators, rarely straying from what I love.',
 'Which of these best describe you?.1_The Thoughtful Selector – I research reviews, ratings, and streaming recommendations before committing.',
 'What time of day do you usually watch content?_Evening',
 'What time of day do you usually watch content?_Late-night',
 'What is your favorite genre?_Comedy & Lighthearted',
 'What is your favorite genre?_Drama & Emotional',
 'What is your favorite genre?_Mystery & Thriller'
                    
                    ]]


movie_attributes = list(movies_df.columns.values)
movie_attributes.remove('Title')
movie_attributes.remove('Round 1?')

user_profiles = {}
for user in user_ratings:
    # Select movies from the base list (the ones the user rated)
    rated_movies = movies_df[movies_df['Title'].isin(user_ratings[user].keys())].copy()
    # Add the user's rating for each movie.
    rated_movies['rating'] = rated_movies['Title'].apply(lambda x: user_ratings[user][x])
    # Compute the weighted average (profile) for this user.
    profile = np.average(rated_movies[movie_attributes].values, axis=0,
                         weights=rated_movies['rating'].values)
    user_profiles[user] = profile

users_encoded = users_df
# For similarity, we use all columns except 'user_id'.
user_feature_columns = [col for col in users_encoded.columns if col != 'user_id']

user_attr_matrix = users_encoded.set_index('user_id')[user_feature_columns].values
user_sim_matrix = cosine_similarity(user_attr_matrix)
user_sim_df = pd.DataFrame(user_sim_matrix, index=users_encoded['user_id'], columns=users_encoded['user_id'])

#similarity matrix makes me think we need more characteristics on here, everyone was close to .99X

def compute_content_score(user_profile, movie_vector):
    """
    Computes cosine similarity between a user profile and a movie's attributes.
    """
    return cosine_similarity(user_profile.reshape(1, -1), movie_vector.reshape(1, -1))[0][0]

alpha = 0.7
# lets play with how we can manipulate this trigger


base_list = ['Harry Potter and the Sorcerer\'s Stone',
'The Dark Knight - Batman Begins', 
'Toy Story', 
'Mean Girls', 
'The Social Network', 
'Get Out',
'Spider-man Into the Spider-verse',
'Spirited Away',
'The Notebook',
'Insidious',
'Inception']

candidate_movies = movies_df[~movies_df['Title'].isin(base_list)].copy()

recommendations = {}

for user in users_df['user_id']:
    content_scores = {}
    collaborative_scores = {}
    
    # Calculate scores for each candidate movie.
    for idx, row in candidate_movies.iterrows():
        movie_title = row['Title']
        movie_vector = row[movie_attributes].values
        
        # --- Content-Based Score ---
        # How similar is this movie to the user's own profile?
        c_score = compute_content_score(user_profiles[user], movie_vector)
        content_scores[movie_title] = c_score
        
        # --- Collaborative Score ---
        # Look at similar users (based on user attributes) and average their content-based scores.
        coll_score_sum = 0
        sim_sum = 0
        for other_user in users_df['user_id']:
            if other_user == user:
                continue
            sim = user_sim_df.loc[user, other_user]
            other_c_score = compute_content_score(user_profiles[other_user], movie_vector)
            coll_score_sum += sim * other_c_score
            sim_sum += sim
        cfb_score = coll_score_sum / sim_sum if sim_sum > 0 else 0
        collaborative_scores[movie_title] = cfb_score
    
    # --- Combine Scores ---
    combined_scores = {}
    for movie_title in content_scores:
        combined = alpha * content_scores[movie_title] + (1 - alpha) * collaborative_scores[movie_title]
        combined_scores[movie_title] = combined
    
    # Pick the candidate movie with the highest combined score.
    best_movie = max(combined_scores, key=combined_scores.get)
    recommendations[user] = {
        'movie': best_movie,
        'combined_score': combined_scores[best_movie],
        'content_score': content_scores[best_movie],
        'collaborative_score': collaborative_scores[best_movie]
    }

for user, rec in recommendations.items():
    print(f"Recommendation for {user}:")
    print(f"  Movie: {rec['movie']}")
    print(f"  Combined Score: {rec['combined_score']:.3f}")
    print(f"  (Content Score: {rec['content_score']:.3f}, Collaborative Score: {rec['collaborative_score']:.3f})")
    print()

