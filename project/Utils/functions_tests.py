import random, json
from NLP_HW.project.get_simple_words import simplify


def get_simple_word_dict(org_sent, target_sent, lexicon, multi_word=0):
	"""
	TODO: Deal with no word in sentence: Pick random shared word. If non exists, pick random.
	TODO: Deal with multiple words in sentence: Pick first or Pick random
	TODO: If multiple simplification - Random by weight - do be dealt with simplify() function
	TODO: Need index in target sentence. If simplification not in there, then pick random.
	:param org_sent:
	:param target_sent:
	:param lexicon:
	:param multi_word:
	:return:
	"""
	potential = []
	org_sent_split = org_sent.split(' ')[:-1]
	target_sent_split = target_sent.split(' ')[:-1]
	for i, word in enumerate(org_sent_split):
		if word in lexicon.keys():
			potential.append((i, word))
	if len(potential) > 1:  # Multiple Potential Words to Simplify
		if multi_word == 0:  # Heuristic - Pick first word in sentence to simplify
			simplified = simplify(potential[0][1], lexicon)  # returns a string (can be multiple words)
			reg_index = potential[0][0]
		else:
			# Option 1 - Random choice with ints
			# rand_ind = random.randint(0, len(potential) - 1)
			# simplified = simplify(potential[rand_ind][1], lexicon)  # returns a string (can be multiple words)
			# reg_index = potential[rand_ind][0]
			# Option 2 - Random choice with choice
			# choice = random.choice(potential)
			# simplified = simplify(choice[1], lexicon)
			# reg_index = choice[0]
			# Option 3 - Random choice with choices (weighted
			ws = [len(lexicon[w[1]]) for w in potential]
			choice = random.choices(potential, ws)
			simplified = simplify(choice[0][1], lexicon)
			reg_index = choice[0][0]
	elif len(potential) == 1:  # One potential word to simplify
		simplified = simplify(potential[0][1], lexicon)  # returns a string (can be multiple words)
		reg_index = potential[0][0]
	else:  # No potential words found
		target_words = set(target_sent_split) if target_sent is not None else set()
		shared_words = (set(org_sent_split) & target_words) - set('.')
		simplified = random.choice(list(shared_words)) if shared_words != set() else random.choice(list(set(org_sent_split) - set('.')))
		reg_index = org_sent_split.index(simplified)
	if target_sent is not None:  # Training time:
		if simplified in target_sent:  # If simplified in sim_sentence, then find index
			sim_index = target_sent_split.index(simplified.split(' ')[0])  # index of first word of simplified
		else:  # else pick random index
			sim_index = random.randint(0, len(target_sent_split) - 1)
		return simplified, reg_index, sim_index  # simplified, index in org_sent, index in target_sent
	else:  # Test time:
		return simplified, reg_index  # simplified, index in org_sent
	
	
if __name__ =='__main__':
	entries = [['דבר שרת המשפטים ', 'דברי פתיחה של שרת המשפטים '], ['בשנת הצטרפה מדינת ישראל לרשימת המדינות שאשררו את אמנת האו ם העוסקת בזכויותיהם של אנשים עם מוגבלויות . ', 'בשנת מדינת ישראל הצטרפה אל כל המדינות שאישרו את אמנת האו ם לזכויות אנשים עם מוגבלות . '], ['מדובר במסמך ייחודי המבסס את התפיסה לפיה אנשים עם מוגבלות זכאים ככל אדם לחירות לכבוד ולשוויון בפני החוק . ', 'האמנה היא מסמך מיוחד . האמנה קובעת שלאנשים עם מוגבלות יש זכות לכבוד לחופש ולשוויון בפני החוק . '], ['בחוברת שלפניכם הסבר על האמנה ערכיה וסעיפיה . ', 'בחוברת הזו יש הסבר על האמנה ועל הכללים שבה . '], ['משרד המשפטים כולו ונציבות שוויון זכויות בתוכו פועלים לקידום חברה שוויונית יותר לאנשים עם מוגבלות בישראל . ', 'במשרד המשפטים ובנציבות שוויון זכויות לאנשים עם מוגבלות הנציבות פועלים כדי לקיים את אמנת האו ם בישראל וכדי שלאנשים עם מוגבלות יהיה שוויון . '], ['המשרד פועל לקידום חקיקה התואמת את ערכי האמנה ובכלל זה תיקון מקיף לחוק הכשרות המשפטית אשר הונח על שולחן הכנסת . ', 'משרד המשפטים פועל כדי לקדם חוקים שקשורים לזכויות . '], [' ', 'נציבות שוויון זכויות לאנשים עם מוגבלות היא חלק ממשרד המשפטים . '], ['נמשיך להוביל בדרך זו ליצירת חברה שוויונית יותר ולמניעת הפלייתם של אנשים עם מוגבלות . ', 'נמשיך לפעול לחברה שווה יותר ולכך שלא תהיה הפליה של אנשים עם מוגבלות . '], ['ח כ איילת שקד שרת המשפטים ', 'איילת שקד שרת המשפטים '], ['דבר מנכ לית משרד המשפטים ', 'דברי פתיחה של המנהלת הכללית של משרד המשפטים '], ['מפלים בין אדם לאדם . . . עליו להיות אדם ובכך אנו אומרים די . ', ' '], ['בין דפיה של האמנה בדבר זכויותיהם של אנשים עם מוגבלות טמונה אחת מאבני היסוד של חזון המדינה ומשרד המשפטים עשיית צדק ושוויון בפני החוק . ', 'באמנה לזכויות אנשים עם מוגבלות מופיעים דברים חשובים שהמדינה ומשרד המשפטים מאמינים בהם לעשות צדק שלא תהיה הפליה ושכולם יהיו שווים בפני החוק . '], ['מאז חתמה ישראל על האמנה בשנת הובילה נציבות שוויון זכויות לאנשים עם מוגבלות בשיתוף מחלקת ייעוץ וחקיקה משפט בין לאומי את הטמעת האמנה והתאמת הדין בישראל לעקרונותיה . ', 'ישראל חתמה על האמנה בשנת . משנת נציבות שוויון זכויות לאנשים עם מוגבלות ומשרד המשפטים עוסקים בקיום האמנה וההתאמה של החוקים בישראל למה שכתוב בה . '], ['בשנת בזכות עבודה אינטנסיבית ומאומצת אושררה האמנה . ', 'ישראל אישרה את האמנה בשנת . '], ['הענקת שוויון מהותי לאנשים עם מוגבלות אינה מסתכמת באשרור האמנה ובאיסור הפליה . ', 'אבל כדי שיהיה שוויון לא מספיק לאשר את האמנה ולדאוג שלא תהיה הפליה שלא יתנהגו אל אנשים עם מוגבלות בצורה שונה מאשר אל אנשים אחרים . '], ['על כולנו מוטלת החובה לפעול באופן אקטיבי על מנת להעניק לאנשים עם מוגבלות נגישות בכל תחומי החיים שתאפשר את שילובם הלכה ולמעשה . ', 'כולנו צריכים לעשות דברים כדי לתת לאנשים עם מוגבלות נגישות לכל דבר בחיים ולאפשר להם להשתלב בחברה בצורה שווה כמו כולם . '], ['משרד המשפטים ימשיך לחתור להשגת חברה שוויונית שבה ניתנת לאנשים עם מוגבלות האפשרות ליהנות הנאה מלאה מזכויותיהם היסודיות . ', 'משרד המשפטים ימשיך לעשות מאמצים כדי שאנשים עם מוגבלות יוכלו ליהנות מהזכויות שלהם כמו כולם . '], ['המדריך שלהלן המפרט ומבהיר את עקרונות האמנה הוא צעד נוסף בדרך להטמעת שוויון הזכויות לאנשים עם מוגבלות . ', 'החוברת מסבירה את הכללים של האמנה . החוברת היא עוד צעד בדרך לשוויון זכויות לאנשים עם מוגבלות . '], ['עו ד אמי פלמור מנכ לית משרד המשפטים ', 'אמי פלמור המנהלת הכללית מנכ לית של משרד המשפטים '], ['דבר נציב שוויון זכויות לאנשים עם מוגבלות ', 'דברי פתיחה של נציב שוויון זכויות לאנשים עם מוגבלות ']]
	data_dir = 'C:\\Users\\eytanc\\OneDrive\\Documents\\University\\2018-9\\Sem B\\NLP\\Project\\Dataset\\'
	with open(data_dir+'lexicon.json', 'r', encoding='utf-8') as f:
		lexicon = json.load(f)
	for en in entries:
		if en[0] != ' ' and en[1] != ' ':
			print(en[0].split(' '))
			print(en[1].split(' '))
			print(get_simple_word_dict(en[0], en[1], lexicon, multi_word=1))
			print()
