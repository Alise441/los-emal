"""Everyday situations a newcomer in Zurich actually runs into."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    id: str
    emoji: str
    title: dict          # by UI language code: en, de
    goal: dict           # what the learner tries to achieve, by UI language
    setting: str         # for the model, German: the place and what the model's character knows
    role: str            # who the model plays, German
    learner: str         # for the model, German, third person: who the learner is and what they want
    voice: str = "f"     # f | m, picks the reference voice
    avatar: str = "🙂"   # shown next to the character's lines
    who: dict = field(default_factory=dict)  # short name of the character, by UI language
    address: str = "Sie"  # how the character addresses the learner: Sie | du
    steps: str = ""       # for the model, German: how the conversation goes, step by step
    # A sample conversation for the model, written and checked by hand: the character's lines (Zurich German)
    # and the learner's (Standard German) take turns, starting with the prepared first line. It shows tone,
    # dialect and course of the scene; Apertus imitated it far better than it followed rules.
    example: tuple = ()
    # The same conversation in Standard German, for DIALOGUE=de (Apertus writes Standard German, then it is translated).
    example_de: tuple = ()


SCENARIOS: list[Scenario] = [
    Scenario(
        "baeckerei", "🥐",
        {"en": "Bakery in the morning", "de": "Bäckerei am Morgen"},
        {"en": "Buy two croissants and a bread, then pay.",
         "de": "Kaufe zwei Gipfeli und ein Brot und bezahle."},
        "Kleine Quartierbäckerei in Zürich-Wiedikon, 7:30 Uhr, es ist viel los. Ein Gipfeli kostet 1.60 Franken, "
        "ein Ruchbrot 4.20 Franken, zwei Gipfeli und ein Ruchbrot zusammen 7.40 Franken; man zahlt bar, mit Karte "
        "oder Twint. Typisch: «Was dörfs sii?», «Suscht no öppis?», «Das macht 7.40.»",
        "Verkäuferin hinter der Theke",
        "Eine Kundin oder ein Kunde, neu in Zürich, möchte zwei Gipfeli und ein Brot kaufen und bezahlen.",
        address="Sie",
        steps=("die Bestellung aufnehmen, fragen, ob noch etwas dazukommt, den Preis nennen, kassieren (bar, Karte "
               "oder Twint), sich verabschieden."),
        example=(
            "Grüezi! Was dörfs sii?",
            "Zwei Gipfeli und ein Ruchbrot, bitte.",
            "Gärn. Suscht no öppis?",
            "Nein, das ist alles.",
            "Das macht 7.40. Zahled Sie bar oder mit Charte?",
            "Mit Karte, bitte.",
            "Sie chönd d Charte da anehebe. … Merci, das hät klappet.",
            "Danke schön.",
            "Merci vilmal und en schöne Tag no!",
        ),
        example_de=(
            "Grüezi! Was darf es sein?",
            "Zwei Gipfeli und ein Ruchbrot, bitte.",
            "Gern. Sonst noch etwas?",
            "Nein, das ist alles.",
            "Das macht 7.40. Zahlen Sie bar oder mit Karte?",
            "Mit Karte, bitte.",
            "Sie können die Karte hier hinhalten. … Danke, das hat geklappt.",
            "Danke schön.",
            "Vielen Dank und noch einen schönen Tag!",
        ),
        avatar="👩‍🍳", who={"en": "The baker", "de": "Die Verkäuferin"},
    ),
    Scenario(
        "migros", "🛒",
        {"en": "Checkout at Migros", "de": "Kasse in der Migros"},
        {"en": "Pay for your groceries. You have no Cumulus card and need a bag.",
         "de": "Bezahle deinen Einkauf. Du hast keine Cumulus-Karte und brauchst eine Tasche."},
        "Migros-Filiale am Limmatplatz, Kasse mit Bedienung. Der Einkauf kostet 23.80 Franken, mit einer "
        "Papiertasche (50 Rappen) 24.30 Franken. Mit der Cumulus-Karte sammelt man Punkte. "
        "Typisch: «Händ Sie e Cumulus-Charte?», «Bruuched Sie es Säckli?», «Mit Charte oder bar?»",
        "Kassierer",
        "Eine Kundin oder ein Kunde, neu in Zürich, möchte den Einkauf bezahlen, hat keine Cumulus-Karte und "
        "braucht eine Tasche.",
        voice="m",
        address="Sie",
        steps=("nach der Cumulus-Karte und dem Säckli fragen, den Betrag nennen, kassieren, den Kassenzettel "
               "anbieten, sich verabschieden."),
        example=(
            "Grüezi! Händ Sie e Cumulus-Charte? Bruuched Sie es Säckli?",
            "Nein, eine Cumulus-Karte habe ich nicht. Aber eine Tasche brauche ich, bitte.",
            "Kei Problem. Mit em Säckli macht das 24.30.",
            "Kann ich mit Karte bezahlen?",
            "Sicher, hebed Sie d Charte eifach as Gerät.",
            "So, erledigt.",
            "Merci. Wänd Sie de Kassezettel?",
            "Nein, danke.",
            "Guet. Merci und en schöne Tag!",
        ),
        example_de=(
            "Grüezi! Haben Sie eine Cumulus-Karte? Brauchen Sie eine Tüte?",
            "Nein, eine Cumulus-Karte habe ich nicht. Aber eine Tasche brauche ich, bitte.",
            "Kein Problem. Mit der Tüte macht das 24.30.",
            "Kann ich mit Karte bezahlen?",
            "Sicher, halten Sie die Karte einfach ans Gerät.",
            "So, erledigt.",
            "Danke. Möchten Sie den Kassenzettel?",
            "Nein, danke.",
            "Gut. Danke und einen schönen Tag!",
        ),
        avatar="👨‍🦱", who={"en": "The cashier", "de": "Der Kassierer"},
    ),
    Scenario(
        "kreisbuero", "🏛️",
        {"en": "Registering at the Kreisbüro", "de": "Anmeldung im Kreisbüro"},
        {"en": "Register your new address. Find out which documents you need.",
         "de": "Melde deine neue Adresse an. Finde heraus, welche Dokumente du brauchst."},
        "Kreisbüro der Stadt Zürich, Schalter für Zuzüge. Für die Anmeldung braucht es den Pass oder die ID und den "
        "Mietvertrag, bei Personen aus dem Ausland auch den Arbeitsvertrag und ein Passfoto. Man meldet sich "
        "innert 14 Tagen nach dem Umzug an.",
        "Sachbearbeiterin am Schalter",
        "Die Person ist neu nach Zürich gezogen und möchte ihre Adresse anmelden. Sie weiss noch nicht, welche "
        "Dokumente sie braucht.",
        address="Sie",
        steps=("das Anliegen klären, nach den Dokumenten fragen, sagen, was noch fehlt und bis wann, sich "
               "verabschieden."),
        example=(
            "Grüezi! Was chan ich für Sie mache?",
            "Ich bin neu nach Zürich gezogen und möchte mich anmelden.",
            "Herzlich willkomme! Händ Sie Ihre Pass oder d ID und de Mietvertrag debii?",
            "Ja, hier sind mein Pass und der Mietvertrag.",
            "Merci. Chömed Sie us em Usland?",
            "Ja, aus Spanien.",
            "Dänn bruuch ich no Ihre Arbeitsvertrag und es Passfoto.",
            "Die habe ich heute nicht dabei.",
            "Kei Problem, Sie händ 14 Täg Ziit. Bringed Sie die eifach spöter verbii. Uf Widerluege!",
        ),
        example_de=(
            "Grüezi! Was kann ich für Sie tun?",
            "Ich bin neu nach Zürich gezogen und möchte mich anmelden.",
            "Herzlich willkommen! Haben Sie Ihren Pass oder die ID und den Mietvertrag dabei?",
            "Ja, hier sind mein Pass und der Mietvertrag.",
            "Danke. Kommen Sie aus dem Ausland?",
            "Ja, aus Spanien.",
            "Dann brauche ich noch Ihren Arbeitsvertrag und ein Passfoto.",
            "Die habe ich heute nicht dabei.",
            "Kein Problem, Sie haben 14 Tage Zeit. Bringen Sie sie einfach später vorbei. Auf Wiedersehen!",
        ),
        avatar="👩‍💼", who={"en": "The clerk", "de": "Die Sachbearbeiterin"},
    ),
    Scenario(
        "arzt", "🩺",
        {"en": "Calling the GP's practice", "de": "Telefon mit der Hausarztpraxis"},
        {"en": "Get an appointment this week because of a sore throat.",
         "de": "Bekomme diese Woche einen Termin wegen Halsschmerzen."},
        "Hausarztpraxis in Oerlikon, Telefonanruf am Montagmorgen. Freie Termine diese Woche: Dienstag um 10.15 Uhr "
        "und Donnerstag um 16.30 Uhr. Neue Patientinnen und Patienten bringen die Krankenkassenkarte mit.",
        "Medizinische Praxisassistentin am Telefon",
        "Die Person ruft an, weil sie Halsschmerzen hat, und möchte diese Woche einen Termin.",
        address="Sie",
        steps=("nach dem Grund fragen, die freien Termine anbieten, einen Termin abmachen, nach dem Namen fragen, "
               "an die Krankenkassenkarte erinnern, gute Besserung wünschen."),
        example=(
            "Praxis Doktor Meier, grüezi! Wie chan ich Ihne hälfe?",
            "Guten Tag, ich habe Halsschmerzen und hätte gern diese Woche einen Termin.",
            "Oh, das tuet mer leid. Mir händ am Ziischtig am Viertel ab zäh oder am Donnschtig am halbi füfi no öppis frei.",
            "Dienstag um Viertel nach zehn passt gut.",
            "Guet. Sind Sie scho Patient bi eus?",
            "Nein, ich komme zum ersten Mal.",
            "Dänn nämed Sie bitte d Chrankekasse-Charte mit. Wie heissed Sie?",
            "Ich heisse Anna Müller.",
            "Merci, Frau Müller. Bis am Ziischtig und gueti Besserig!",
        ),
        example_de=(
            "Praxis Doktor Meier, grüezi! Wie kann ich Ihnen helfen?",
            "Guten Tag, ich habe Halsschmerzen und hätte gern diese Woche einen Termin.",
            "Oh, das tut mir leid. Wir haben am Dienstag um Viertel nach zehn oder am Donnerstag um halb fünf noch etwas frei.",
            "Dienstag um Viertel nach zehn passt gut.",
            "Gut. Sind Sie schon Patient bei uns?",
            "Nein, ich komme zum ersten Mal.",
            "Dann nehmen Sie bitte die Krankenkassenkarte mit. Wie heissen Sie?",
            "Ich heisse Anna Müller.",
            "Danke, Frau Müller. Bis Dienstag und gute Besserung!",
        ),
        avatar="👩‍⚕️", who={"en": "The practice assistant", "de": "Die Praxisassistentin"},
    ),
    Scenario(
        "nachbarin", "🏠",
        {"en": "Meeting the neighbour", "de": "Die Nachbarin im Treppenhaus"},
        {"en": "Introduce yourself and ask how the laundry room schedule works.",
         "de": "Stell dich vor und frag, wie der Waschplan funktioniert."},
        "Treppenhaus eines Mehrfamilienhauses in Zürich-Altstetten. Die Waschküche ist im Keller; laut Waschplan "
        "hat jede Wohnung einen festen Waschtag, die neue Wohnung im 2. Stock den Mittwoch. Nach dem Trocknen "
        "putzt man das Sieb vom Tumbler.",
        "Ältere Nachbarin, freundlich, wohnt seit 30 Jahren im Haus",
        "Die Person ist neu im Haus eingezogen, möchte sich vorstellen und fragt, wie der Waschplan funktioniert.",
        address="Sie",
        steps=("sich vorstellen, die neue Person im Haus willkommen heissen, ihre Fragen zum Waschplan beantworten "
               "(Waschtag, Sieb vom Tumbler), sich verabschieden."),
        example=(
            "Grüezi! Ich bi d Martha, ich wohne scho drissg Jahr da im Huus. Sie sind sicher neu da?",
            "Ja, ich bin letzte Woche in den zweiten Stock eingezogen. Ich heisse Sam.",
            "Herzlich willkomme im Huus! Wänn Sie öppis bruuched, lüüted Sie eifach bi mir.",
            "Danke! Wie funktioniert das mit der Waschküche?",
            "D Wöschchuchi isch im Chäller, und jedi Wohnig hät ihre Wöschtag. Sie händ de Mittwuch.",
            "Mittwoch, gut. Muss ich sonst noch etwas wissen?",
            "Nach em Tröchne putzed Sie bitte s Sieb vom Tumbler. Dänn sind alli zfriede.",
            "Alles klar, vielen Dank!",
            "Gärn gscheh! En schöne Tag no, Sam.",
        ),
        example_de=(
            "Grüezi! Ich bin die Martha, ich wohne schon dreissig Jahre hier im Haus. Sie sind sicher neu hier?",
            "Ja, ich bin letzte Woche in den zweiten Stock eingezogen. Ich heisse Sam.",
            "Herzlich willkommen im Haus! Wenn Sie etwas brauchen, läuten Sie einfach bei mir.",
            "Danke! Wie funktioniert das mit der Waschküche?",
            "Die Waschküche ist im Keller, und jede Wohnung hat ihren Waschtag. Sie haben den Mittwoch.",
            "Mittwoch, gut. Muss ich sonst noch etwas wissen?",
            "Nach dem Trocknen putzen Sie bitte das Sieb vom Tumbler. Dann sind alle zufrieden.",
            "Alles klar, vielen Dank!",
            "Gern geschehen! Noch einen schönen Tag, Sam.",
        ),
        avatar="👵", who={"en": "Your neighbour", "de": "Deine Nachbarin"},
    ),
    Scenario(
        "buero", "☕",
        {"en": "Coffee break at work", "de": "Znüni-Pause im Büro"},
        {"en": "Small talk with a colleague about the weekend.",
         "de": "Small Talk mit einem Kollegen über das Wochenende."},
        "Kaffeeküche in einem Büro in Zürich-West, Montagmorgen. Man duzt sich. Du warst am Samstag auf dem "
        "Uetliberg wandern und am Sonntag am Match vom FC Zürich.",
        "Arbeitskollege",
        "Die Person ist eine neue Kollegin oder ein neuer Kollege und plaudert in der Pause über das Wochenende.",
        voice="m",
        address="du",
        steps=("vom eigenen Wochenende erzählen und beim Gegenüber nachfragen, ein paar Mal hin und her, dann "
               "zurück an die Arbeit."),
        example=(
            "Hoi! Was häsch du übers Wuchenänd gmacht?",
            "Ich war am Samstag in der Stadt. Und du?",
            "Ich bi am Samschtig uf em Üetliberg go wandere. Bi dem Wätter isch das mega schön gsi.",
            "Schön! Und am Sonntag?",
            "Am Sunntig bin ich am Match vom FCZ gsi. S isch es super Spiel gsi.",
            "Ich war noch nie bei einem Fussballspiel in Zürich.",
            "Dänn chunnsch s nächscht Mal mit! So, ich mues wieder go schaffe. Bis spöter!",
        ),
        example_de=(
            "Hallo! Was hast du übers Wochenende gemacht?",
            "Ich war am Samstag in der Stadt. Und du?",
            "Ich war am Samstag auf dem Uetliberg wandern. Bei dem Wetter war das sehr schön.",
            "Schön! Und am Sonntag?",
            "Am Sonntag war ich am Match vom FCZ. Es war ein super Spiel.",
            "Ich war noch nie bei einem Fussballspiel in Zürich.",
            "Dann kommst du das nächste Mal mit! So, ich muss wieder arbeiten gehen. Bis später!",
        ),
        avatar="👨‍💻", who={"en": "Your colleague", "de": "Dein Arbeitskollege"},
    ),
    Scenario(
        "wohnung", "🔑",
        {"en": "Flat viewing", "de": "Wohnungsbesichtigung"},
        {"en": "Ask about rent, extra costs and the move-in date.",
         "de": "Frag nach Miete, Nebenkosten und dem Einzugstermin."},
        "Besichtigung einer 3-Zimmer-Wohnung in Zürich-Wipkingen, viele Interessenten. Miete 2'350 Franken plus "
        "180 Franken Nebenkosten im Monat, frei ab 1. Dezember. Für die Bewerbung braucht es das Formular, einen "
        "Betreibungsauszug und eine Ausweiskopie.",
        "Bewirtschafterin der Liegenschaftsverwaltung",
        "Die Person interessiert sich für die Wohnung und möchte nach Miete, Nebenkosten und Einzugstermin fragen.",
        address="Sie",
        steps=("die Fragen zu Miete, Nebenkosten und Einzug beantworten, die Bewerbung erklären, sich "
               "verabschieden."),
        example=(
            "Grüezi, chömed Sie ine! Ich bi d Frau Keller vo de Verwaltig. Händ Sie Frage zur Wohnig?",
            "Ja, wie hoch ist die Miete?",
            "D Miete isch 2'350 Franke im Monet, plus 180 Franke Nebechöschte.",
            "Und ab wann ist die Wohnung frei?",
            "Ab em erschte Dezämber.",
            "Wie kann ich mich bewerben?",
            "Fülled Sie s Formular uus und schicked Sie eus en Betriibigsuuszug und e Kopie vom Uswiis.",
            "Gut, das mache ich. Vielen Dank!",
            "Gärn gscheh. Uf Widerluege!",
        ),
        example_de=(
            "Grüezi, kommen Sie herein! Ich bin Frau Keller von der Verwaltung. Haben Sie Fragen zur Wohnung?",
            "Ja, wie hoch ist die Miete?",
            "Die Miete ist 2'350 Franken im Monat, plus 180 Franken Nebenkosten.",
            "Und ab wann ist die Wohnung frei?",
            "Ab dem ersten Dezember.",
            "Wie kann ich mich bewerben?",
            "Füllen Sie das Formular aus und schicken Sie uns einen Betreibungsauszug und eine Kopie vom Ausweis.",
            "Gut, das mache ich. Vielen Dank!",
            "Gern geschehen. Auf Wiedersehen!",
        ),
        avatar="👩‍💼", who={"en": "The property manager", "de": "Die Bewirtschafterin"},
    ),
    Scenario(
        "restaurant", "🍽️",
        {"en": "Restaurant in the old town", "de": "Restaurant in der Altstadt"},
        {"en": "Order a meal and a drink, later pay separately.",
         "de": "Bestelle ein Essen und ein Getränk, später getrennt bezahlen."},
        "Traditionelles Restaurant im Niederdorf, Mittagszeit. Tagesmenü: Zürcher Geschnetzeltes mit Rösti "
        "34 Franken oder Älplermagronen 26 Franken; Mineral oder Rivella 5.50, ein Glas Weisswein 8 Franken. "
        "Typisch: «Was dörfs sii?», «Und zum Trinke?», «Zäme oder separat?»",
        "Kellnerin",
        "Die Person isst mit einer Freundin zu Mittag, möchte ein Essen und ein Getränk bestellen und am Schluss "
        "getrennt bezahlen.",
        address="Sie",
        steps=("Essen und Getränk aufnehmen, später fragen, ob zusammen oder separat bezahlt wird, den Betrag "
               "nennen, kassieren, sich verabschieden."),
        example=(
            "Grüezi mitenand! Was dörfs sii zum Zmittag?",
            "Ich nehme die Älplermagronen, bitte.",
            "Gärn. Und zum Trinke?",
            "Ein Rivella, bitte.",
            "Chunnt sofort.",
            "Wir möchten bitte bezahlen.",
            "Gärn. Zäme oder separat?",
            "Separat, bitte.",
            "Für Sie macht das 31.50.",
            "Ich zahle mit Karte.",
            "Merci vilmal und en schöne Namittag!",
        ),
        example_de=(
            "Grüezi miteinander! Was darf es zum Mittagessen sein?",
            "Ich nehme die Älplermagronen, bitte.",
            "Gern. Und zu trinken?",
            "Ein Rivella, bitte.",
            "Kommt sofort.",
            "Wir möchten bitte bezahlen.",
            "Gern. Zusammen oder separat?",
            "Separat, bitte.",
            "Für Sie macht das 31.50.",
            "Ich zahle mit Karte.",
            "Vielen Dank und einen schönen Nachmittag!",
        ),
        avatar="💁‍♀️", who={"en": "The waitress", "de": "Die Kellnerin"},
    ),
    Scenario(
        "tram", "🚋",
        {"en": "At the tram stop", "de": "An der Tramhaltestelle"},
        {"en": "An older lady asks you something. Help her if you can.",
         "de": "Eine ältere Dame fragt dich etwas. Hilf ihr, wenn du kannst."},
        "Tramhaltestelle Central, Abend, es regnet leicht. Die Dame will ins Kunsthaus, kennt den Weg aber nicht "
        "(richtig wäre: Tram 3 Richtung Klusplatz bis zur Haltestelle Kunsthaus, oder zu Fuss etwa 10 Minuten).",
        "Ältere Dame, die den Weg zum Kunsthaus sucht",
        "Die Person wartet an der Haltestelle. Die Dame spricht sie an und fragt nach dem Weg; die Person hilft, "
        "so gut sie kann.",
        address="Sie",
        steps=("nach dem Weg zum Kunsthaus fragen, nachfragen (welches Tram, wo aussteigen, wie weit zu Fuss), sich "
               "bedanken und verabschieden."),
        example=(
            "Exgüsi, chönd Sie mir säge, wie ich zum Kunsthuus chume? Ich bi no nie dört gsi.",
            "Ja, Sie können das Tram Nummer 3 Richtung Klusplatz nehmen.",
            "S Drüü, aha. Und wo mues ich uusstiige?",
            "An der Haltestelle Kunsthaus, nach ein paar Stationen.",
            "Merci. Chönnt ich au z Fuess gah?",
            "Ja, zu Fuss sind es etwa zehn Minuten.",
            "Bi dem Räge nimm ich lieber s Tram. Merci vilmal, Sie sind e grossi Hilf! Uf Widerluege!",
        ),
        example_de=(
            "Entschuldigung, können Sie mir sagen, wie ich zum Kunsthaus komme? Ich war noch nie dort.",
            "Ja, Sie können das Tram Nummer 3 Richtung Klusplatz nehmen.",
            "Die Drei, aha. Und wo muss ich aussteigen?",
            "An der Haltestelle Kunsthaus, nach ein paar Stationen.",
            "Danke. Könnte ich auch zu Fuss gehen?",
            "Ja, zu Fuss sind es etwa zehn Minuten.",
            "Bei dem Regen nehme ich lieber das Tram. Vielen Dank, Sie sind eine grosse Hilfe! Auf Wiedersehen!",
        ),
        avatar="👵", who={"en": "An older lady", "de": "Eine ältere Dame"},
    ),
]

# Not shown in the app any more (its ticket words were the hardest for the translators to get right); kept because
# the E4 test set, data/learner_de_testset.jsonl, has 6 items from this scene.
EVAL_ONLY: list[Scenario] = [
    Scenario(
        "sbb", "🚆",
        {"en": "SBB ticket counter", "de": "SBB-Schalter"},
        {"en": "Buy a return ticket to Lucerne for tomorrow. You have a Halbtax.",
         "de": "Kaufe ein Retourbillett nach Luzern für morgen. Du hast ein Halbtax."},
        "SBB Reisezentrum im Hauptbahnhof Zürich. Ein Retourbillett Zürich–Luzern kostet mit Halbtax in der "
        "2. Klasse 27 Franken, in der 1. Klasse 47 Franken, ohne Halbtax das Doppelte; die Züge fahren jede halbe "
        "Stunde. Typisch: «Eifach oder retour?» (eifach = nur Hinfahrt), «Erschti oder zweiti Klass?», "
        "«Händ Sie es Halbtax?»",
        "Mitarbeiter am Schalter",
        "Die Person möchte ein Retourbillett nach Luzern für morgen kaufen. Sie hat ein Halbtax.",
        voice="m",
        address="Sie",
        steps=("Ziel und Datum klären, eifach oder retour, Halbtax, Klasse, den Preis nennen, kassieren, eine gute "
               "Reise wünschen."),
        example=(
            "Grüezi, wie chan ich Ihne hälfe?",
            "Ich möchte ein Billett nach Luzern für morgen.",
            "Gärn. Eifach oder retour?",
            "Retour, bitte.",
            "Händ Sie es Halbtax?",
            "Ja, ich habe ein Halbtax.",
            "Erschti oder zweiti Klass?",
            "Zweite Klasse.",
            "Das macht 27 Franke. D Züg fahred jedi halbi Stund.",
            "Ich zahle mit Karte.",
            "Merci, da isch Ihres Billett. Gueti Reis!",
        ),
        example_de=(
            "Grüezi, wie kann ich Ihnen helfen?",
            "Ich möchte ein Billett nach Luzern für morgen.",
            "Gern. Nur Hinfahrt oder retour?",
            "Retour, bitte.",
            "Haben Sie ein Halbtax?",
            "Ja, ich habe ein Halbtax.",
            "Erste oder zweite Klasse?",
            "Zweite Klasse.",
            "Das macht 27 Franken. Die Züge fahren jede halbe Stunde.",
            "Ich zahle mit Karte.",
            "Danke, hier ist Ihr Billett. Gute Reise!",
        ),
        avatar="👨‍💼", who={"en": "The SBB clerk", "de": "Der SBB-Mitarbeiter"},
    ),
]

BY_ID = {s.id: s for s in SCENARIOS + EVAL_ONLY}

