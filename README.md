# Speech_Pattern_Bot
Discord bot built on an NLP that recognizes speech patterns of a certain user and flags messages with similar patterns. Started as a joke between me and my friends, turned into a fun project.

> [!NOTE]
> The accuracy on this model may vary on the size of training data, relative size of datasets, quality of data and other factors.

# Dependencies 
`pip install -r requirements.txt
`
- pandas
- joblib
- scikit-learn
- discord
  
# Setup Steps
## 1. Export chat(s)
   Prefferably with that has a large amount of messages from the user you want to pick up the speech patterns of. <br>
   Use: https://github.com/Tyrrrz/DiscordChatExporter
   if you want the formatting python file under /data to work
   
## 2. Change all the placeholders 
  - File directories under csvCreation.py
  - File directories under NLP.py
  - Message to respond with (line 62 Bot.py)
  - Bot Token in Bot.py (line 129)
  

## 3. Configure Bot under Discord <a href="https://discord.com/developers/applications">Developer Portal</a>

## 4. Run bot.py
<br>
<img width="587" alt="Screenshot 2024-08-09 at 11 49 37 PM" src="https://github.com/user-attachments/assets/9bac64ca-5156-4c1c-97fe-4b97a69a1d1f">
<br>

#### 5. (Optional) Change threshold in Bot.py to change 'strictness' of bot

# Usage
When running, the bot will automatically send a message with whatever you set it to, in the case of my friend who speaks very 'methodologically' it will reply with 'Methodicall'. The bot will then give you the option to reply with a check () or a cross () upon which it will expand the data set and the new accuracy data will output to the console.

<img width="425" alt="Screenshot 2024-08-09 at 11 48 43 PM" src="https://github.com/user-attachments/assets/9337815f-f03b-46c1-a98a-fcc37ed2f10a">

<br>

***

You can  use the application command `/Accuracy` to return the models accuracy to the channel where it is used.

<img width="562" alt="Screenshot 2024-08-09 at 11 54 05 PM" src="https://github.com/user-attachments/assets/8a98cb8a-aab3-4189-81fe-e818a9c85d6e">

<br>

---
Soon: 
- [ ] Automate the downloading all required data through 1 app command. 
- [ ] Command to turn bot 'on' and 'off'.
- [ ] Command to change 'harshness' of bot.
