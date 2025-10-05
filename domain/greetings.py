from domain.enums import Gender, AgeBucket

def greet(gender: Gender, bucket: AgeBucket) -> str:
    if bucket == AgeBucket.u18:
        return "Привет, юный джентльмен!" if gender == Gender.male else "Привет, юная леди!"
    if bucket == AgeBucket._18_30:
        return "Здравствуй, молодой человек!" if gender == Gender.male else "Здравствуй, молодая девушка!"
    if bucket == AgeBucket._30_40:
        return "Приветствую, уважаемый мужчина!" if gender == Gender.male else "Приветствую, уважаемая женщина!"
    return "Добрый день, опытный джентльмен!" if gender == Gender.male else "Добрый день, опытная леди!"