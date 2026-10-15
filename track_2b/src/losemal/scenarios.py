"""Everyday situations a newcomer in Zurich actually runs into."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    emoji: str
    title: dict          # by UI language code: en, de, ru
    goal: dict           # what the learner tries to achieve, by UI language
    setting: str         # for the model, German: the place and what the model's character knows
    role: str            # who the model plays, German
    learner: str         # for the model, German, third person: who the learner is and what they want
    voice: str = "f"     # f | m, picks the reference voice


SCENARIOS: list[Scenario] = [
    Scenario(
        "baeckerei", "🥐",
        {"en": "Bakery in the morning", "de": "Bäckerei am Morgen", "ru": "Пекарня утром"},
        {"en": "Buy two croissants and a bread, then pay.",
         "de": "Kaufe zwei Gipfeli und ein Brot und bezahle.",
         "ru": "Купить два круассана и хлеб, потом заплатить."},
        "Kleine Quartierbäckerei in Zürich-Wiedikon, 7:30 Uhr, es ist viel los. Ein Gipfeli kostet 1.60 Franken, "
        "ein Ruchbrot 4.20 Franken, zwei Gipfeli und ein Ruchbrot zusammen 7.40 Franken; man zahlt bar, mit Karte "
        "oder Twint. Typisch: «Was dörfs sii?», «Suscht no öppis?», «Das macht 7.40.»",
        "Verkäuferin hinter der Theke",
        "Eine Kundin oder ein Kunde, neu in Zürich, möchte zwei Gipfeli und ein Brot kaufen und bezahlen.",
    ),
    Scenario(
        "migros", "🛒",
        {"en": "Checkout at Migros", "de": "Kasse in der Migros", "ru": "Касса в Migros"},
        {"en": "Pay for your groceries. You have no Cumulus card and need a bag.",
         "de": "Bezahle deinen Einkauf. Du hast keine Cumulus-Karte und brauchst eine Tasche.",
         "ru": "Оплатить покупки. Карты Cumulus у тебя нет, нужен пакет."},
        "Migros-Filiale am Limmatplatz, Kasse mit Bedienung. Der Einkauf kostet 23.80 Franken, mit einer "
        "Papiertasche (50 Rappen) 24.30 Franken. Mit der Cumulus-Karte sammelt man Punkte. "
        "Typisch: «Händ Sie e Cumulus-Charte?», «Bruuched Sie es Säckli?», «Mit Charte oder bar?»",
        "Kassierer",
        "Eine Kundin oder ein Kunde, neu in Zürich, möchte den Einkauf bezahlen, hat keine Cumulus-Karte und "
        "braucht eine Tasche.",
        voice="m",
    ),
    Scenario(
        "kreisbuero", "🏛️",
        {"en": "Registering at the Kreisbüro", "de": "Anmeldung im Kreisbüro", "ru": "Регистрация в Kreisbüro"},
        {"en": "Register your new address. Find out which documents you need.",
         "de": "Melde deine neue Adresse an. Finde heraus, welche Dokumente du brauchst.",
         "ru": "Зарегистрировать новый адрес и узнать, какие нужны документы."},
        "Kreisbüro der Stadt Zürich, Schalter für Zuzüge. Für die Anmeldung braucht es den Pass oder die ID und den "
        "Mietvertrag, bei Personen aus dem Ausland auch den Arbeitsvertrag und ein Passfoto. Man meldet sich "
        "innert 14 Tagen nach dem Umzug an.",
        "Sachbearbeiterin am Schalter",
        "Die Person ist neu nach Zürich gezogen und möchte ihre Adresse anmelden. Sie weiss noch nicht, welche "
        "Dokumente sie braucht.",
    ),
    Scenario(
        "arzt", "🩺",
        {"en": "Calling the GP's practice", "de": "Telefon mit der Hausarztpraxis", "ru": "Звонок в кабинет врача"},
        {"en": "Get an appointment this week because of a sore throat.",
         "de": "Bekomme diese Woche einen Termin wegen Halsschmerzen.",
         "ru": "Записаться на этой неделе из-за боли в горле."},
        "Hausarztpraxis in Oerlikon, Telefonanruf am Montagmorgen. Freie Termine diese Woche: Dienstag um 10.15 Uhr "
        "und Donnerstag um 16.30 Uhr. Neue Patientinnen und Patienten bringen die Krankenkassenkarte mit.",
        "Medizinische Praxisassistentin am Telefon",
        "Die Person ruft an, weil sie Halsschmerzen hat, und möchte diese Woche einen Termin.",
    ),
    Scenario(
        "sbb", "🚆",
        {"en": "SBB ticket counter", "de": "SBB-Schalter", "ru": "Касса SBB"},
        {"en": "Buy a return ticket to Lucerne for tomorrow. You have a Halbtax.",
         "de": "Kaufe ein Retourbillett nach Luzern für morgen. Du hast ein Halbtax.",
         "ru": "Купить билет туда-обратно в Люцерн на завтра. У тебя есть Halbtax."},
        "SBB Reisezentrum im Hauptbahnhof Zürich. Ein Retourbillett Zürich–Luzern kostet mit Halbtax in der "
        "2. Klasse 27 Franken, in der 1. Klasse 47 Franken, ohne Halbtax das Doppelte; die Züge fahren jede halbe "
        "Stunde. Typisch: «Eifach oder retour?», «Erschti oder zweiti Klass?», «Händ Sie es Halbtax?»",
        "Mitarbeiter am Schalter",
        "Die Person möchte ein Retourbillett nach Luzern für morgen kaufen. Sie hat ein Halbtax.",
        voice="m",
    ),
    Scenario(
        "nachbarin", "🏠",
        {"en": "Meeting the neighbour", "de": "Die Nachbarin im Treppenhaus", "ru": "Соседка на лестнице"},
        {"en": "Introduce yourself and ask how the laundry room schedule works.",
         "de": "Stell dich vor und frag, wie der Waschplan funktioniert.",
         "ru": "Представиться и спросить, как устроено расписание прачечной."},
        "Treppenhaus eines Mehrfamilienhauses in Zürich-Altstetten. Die Waschküche ist im Keller; laut Waschplan "
        "hat jede Wohnung einen festen Waschtag, die neue Wohnung im 2. Stock den Mittwoch. Nach dem Trocknen "
        "putzt man das Sieb vom Tumbler.",
        "Ältere Nachbarin, freundlich, wohnt seit 30 Jahren im Haus",
        "Die Person ist neu im Haus eingezogen, möchte sich vorstellen und fragt, wie der Waschplan funktioniert.",
    ),
    Scenario(
        "buero", "☕",
        {"en": "Coffee break at work", "de": "Znüni-Pause im Büro", "ru": "Перерыв на кофе на работе"},
        {"en": "Small talk with a colleague about the weekend.",
         "de": "Small Talk mit einem Kollegen über das Wochenende.",
         "ru": "Поболтать с коллегой о выходных."},
        "Kaffeeküche in einem Büro in Zürich-West, Montagmorgen. Man duzt sich. Du warst am Samstag auf dem "
        "Uetliberg wandern und am Sonntag am Match vom FC Zürich.",
        "Arbeitskollege",
        "Die Person ist eine neue Kollegin oder ein neuer Kollege und plaudert in der Pause über das Wochenende.",
        voice="m",
    ),
    Scenario(
        "wohnung", "🔑",
        {"en": "Flat viewing", "de": "Wohnungsbesichtigung", "ru": "Просмотр квартиры"},
        {"en": "Ask about rent, extra costs and the move-in date.",
         "de": "Frag nach Miete, Nebenkosten und dem Einzugstermin.",
         "ru": "Спросить про аренду, коммунальные расходы и дату въезда."},
        "Besichtigung einer 3-Zimmer-Wohnung in Zürich-Wipkingen, viele Interessenten. Miete 2'350 Franken plus "
        "180 Franken Nebenkosten im Monat, frei ab 1. Dezember. Für die Bewerbung braucht es das Formular, einen "
        "Betreibungsauszug und eine Ausweiskopie.",
        "Bewirtschafterin der Liegenschaftsverwaltung",
        "Die Person interessiert sich für die Wohnung und möchte nach Miete, Nebenkosten und Einzugstermin fragen.",
    ),
    Scenario(
        "restaurant", "🍽️",
        {"en": "Restaurant in the old town", "de": "Restaurant in der Altstadt", "ru": "Ресторан в старом городе"},
        {"en": "Order a meal and a drink, later pay separately.",
         "de": "Bestelle ein Essen und ein Getränk, später getrennt bezahlen.",
         "ru": "Заказать блюдо и напиток, потом заплатить раздельно."},
        "Traditionelles Restaurant im Niederdorf, Mittagszeit. Tagesmenü: Zürcher Geschnetzeltes mit Rösti "
        "34 Franken oder Älplermagronen 26 Franken; Mineral oder Rivella 5.50, ein Glas Weisswein 8 Franken. "
        "Typisch: «Was dörfs sii?», «Und zum Trinke?», «Zäme oder separat?»",
        "Kellnerin",
        "Die Person isst mit einer Freundin zu Mittag, möchte ein Essen und ein Getränk bestellen und am Schluss "
        "getrennt bezahlen.",
    ),
    Scenario(
        "tram", "🚋",
        {"en": "At the tram stop", "de": "An der Tramhaltestelle", "ru": "На трамвайной остановке"},
        {"en": "An older lady asks you something. Help her if you can.",
         "de": "Eine ältere Dame fragt dich etwas. Hilf ihr, wenn du kannst.",
         "ru": "Пожилая дама о чём-то спрашивает. Помоги, если сможешь."},
        "Tramhaltestelle Central, Abend, es regnet leicht. Die Dame will ins Kunsthaus, kennt den Weg aber nicht "
        "(richtig wäre: Tram 3 Richtung Klusplatz bis zur Haltestelle Kunsthaus, oder zu Fuss etwa 10 Minuten).",
        "Ältere Dame, die den Weg zum Kunsthaus sucht",
        "Die Person wartet an der Haltestelle. Die Dame spricht sie an und fragt nach dem Weg; die Person hilft, "
        "so gut sie kann.",
    ),
]

BY_ID = {s.id: s for s in SCENARIOS}


def label(s: Scenario, ui: str) -> str:
    return f"{s.emoji} {s.title.get(ui, s.title['en'])}"
