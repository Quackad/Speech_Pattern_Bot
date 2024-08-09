import pandas as pd
import joblib
from discord.ext import commands
from discord import app_commands
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report  #Import accuracy_score

#Load model
pipeline = joblib.load('model.pkl')

intents = discord.Intents.default()
intents.message_content = True  
intents.reactions = True      
intents.guilds = True           

bot = commands.Bot(command_prefix='!', intents=intents)

TICK_EMOJI = '✅'
CROSS_EMOJI = '❌'

def preprocess(text):
    text = str(text).lower()
    if text.startswith('https://'):
        return None
    return text

def harshPrediction(text, threshold=0.6):
    if text is None:
        return 0
    text_vec = pipeline.named_steps['tfidfvectorizer'].transform([text])
    probs = pipeline.named_steps['multinomialnb'].predict_proba(text_vec)
    return 1 if probs[0][1] > threshold else 0

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    await bot.tree.sync()

@bot.event
async def on_message(message):
    #Prevent the bot from responding to its own messages
    if message.author == bot.user:
        return

    message_text = preprocess(message.content)
    if message_text is None:
        return

    #Makes sure it doesnt reply to same message multiple times
    async for msg in message.channel.history(limit=10):
        if msg.author == bot.user and msg.reference and msg.reference.message_id == message.id:
            return

    #Make prediction
    prediction = harshPrediction(message_text, threshold=0.6)  # 
    if prediction == 1:
        #Send a message and add reactions to that message
        response = await message.channel.send("MESSAGE TO SEND WHEN POSITIVE")
        await response.add_reaction(TICK_EMOJI)
        await response.add_reaction(CROSS_EMOJI)

    #Ensure that commands are processed
    await bot.process_commands(message)

@bot.event
async def on_reaction_add(reaction, user):
    #Only process reactions on its own message
    if reaction.message.author == bot.user and user != bot.user:
        if reaction.emoji == TICK_EMOJI:
            label = 1
        elif reaction.emoji == CROSS_EMOJI:
            label = 0
        else:
            return

        #Update training data
        message_text = preprocess(reaction.message.content)
        if message_text is not None:
            df = pd.read_csv('data/combinedMessages.csv')
            new_data = pd.DataFrame({'Content': [message_text], 'Label': [label]})
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv('data/combinedMessages.csv', index=False)

            #Retrain the model
            retrain_model()

def retrain_model():
    df = pd.read_csv('data/combinedMessages.csv')
    df['ProcessedContent'] = df['Content'].apply(preprocess)
    df = df.dropna(subset=['ProcessedContent'])
    
    X = df['ProcessedContent']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Train model on new pipeline
    new_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    new_pipeline.fit(X_train, y_train)
    
    #Save new model
    joblib.dump(new_pipeline, 'model.pkl')

    #Print retrained data to terminal
    y_pred = new_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # Use accuracy_score to compute accuracy

    #Print retrained data to terminal
    classification_report_str = classification_report(y_test, y_pred)
    print("Model Retraining Complete")
    print(classification_report_str)

    return accuracy, classification_report_str

@bot.tree.command(name="accuracy", description="Return Current Model Performacne.")
async def accuracy(interaction: discord.Interaction):
    accuracy, report = retrain_model()
    embed = discord.Embed(
        title="Model Performance",
        color=discord.Color.blue()
    )
    embed.add_field(name="Accuracy", value=f"{accuracy:.2%}", inline=False)
    embed.add_field(name="Classification Report", value=f"```\n{report}\n```", inline=False)

    await interaction.response.send_message(embed=embed)

bot.run('DISCORD_TOKEN')
