ODINW_PROMPTS = {
    "AerialMaritimeDrone": lambda name: "a ship" if name == "boat" else name,
    "AmericanSignLanguageLetters": lambda name: "hand gesture '{}'".format(name),
    "BCCD": lambda name: "Red-Blood-Cell-(RBC)"
    if name == "RBC"
    else "White-Blood-Cell-(WBC)"
    if name == "WBC"
    else "Blood-Platelet-Cell-(BPC)"
    if name == "Platelets"
    else name,
    "boggleBoards": lambda name: "letter '{}'".format(name.upper()),
    "brackishUnderwater": lambda name: "big_fish" if name == "fish" else name,
    "ChessPieces": lambda name: "chess piece {}".format(name),
    "dice": lambda name: "dice {}".format(name),
    "DroneControl": lambda name: "body gesture '{}'".format(name),
    "EgoHands-specific": lambda name: "{} hand".format(name),
    # "EgoHands-specific": lambda name: "my left hand"
    # if name == "myleft"
    # else "my right hand"
    # if name == "myright"
    # else "your right hand"
    # if name == "yourright"
    # else "your left hand"
    # if name == "yourleft"
    # else name,
    "HardHatWorkers": lambda name: "human head wearing a helmet"
    if name == "helmet"
    else "human head"
    if name == "head"
    else name,
    "MaskWearing": lambda name: "human head wearing a mask"
    if name == "mask"
    else "human head"
    if name == "no-mask"
    else name,
    "MountainDewCommercial": lambda name: "small {}".format(name),
    "NorthAmericaMushrooms": lambda name: "mushroom {}".format(name),
    "openPoetryVision": lambda name: "some text with font {}".format(name),
    "OxfordPets-by-breed": lambda name: "head of {}".format(name),
    "OxfordPets-by-species": lambda name: "head of {}".format(name),
    "PKLot": lambda name: "{} parking slot".format(name),
    "pothole": lambda name: "broken {}".format(name),
    "ThermalCheetah": lambda name: "person" if name == "human" else name,
    "UnoCards": lambda name: "Arabic numerals 0"
    if name == "0"
    else "Arabic numerals 1"
    if name == "1"
    else "Arabic numerals +4"
    if name == "2"
    else "Arabic numerals +2"
    if name == "3"
    else "two arrows"
    if name == "4"
    else "cross cycle"
    if name == "5"
    else "colorful cycle"
    if name == "6"
    else "Arabic numerals 2"
    if name == "7"
    else "Arabic numerals 3"
    if name == "8"
    else "Arabic numerals 4"
    if name == "9"
    else "Arabic numerals 5"
    if name == "10"
    else "Arabic numerals 6"
    if name == "11"
    else "Arabic numerals 7"
    if name == "12"
    else "Arabic numerals 8"
    if name == "13"
    else "Arabic numerals 9"
    if name == "14"
    else name,
}
