## Схема решения и метрики

### 1) Решение с примером использования.

В качестве решения будет разработан поисковый сервис: пользователь в специальную строку пишет свой запрос, текст запроса обрабатывается и пользователь в ответ получает k объявлений, отранжированные по релевантности запросу.


### 2) Какая бизнес-метрика должна оптимизироваться? Какое её значение будет считаться успехом?

Бизнес-метрика - конверсия в контакт/действие с продавцом:

- в числителе кол-во пользователей, совершивших одно из действий:
    1. клик на значок "Написать";
    1. клик на значок "Показать телефон";
    1. сразу оформление/запрос доставки (но это, пожалуй, будет сильно реже, так как это кнопка есть в меньшем числе объявлений + само действие более близкое к покупке);
- в знаменателе: кол-во пользователей, кто начал/сделал поиск.

В идеале решение нужно сравнивать с предыдущим лучшим, если его нет, то с каким-нибудь бейзлайном, который легко и быстро имплементировать (косинусное расстояние на TF-IDF векторах + сортировка по популярности/кол-ву покупок). Сравнивать можно с помощью АБ теста (стратифицировать по категориям товаров + может географически).


### 3) Какая метрика машинного обучения будет наилучшим образом отражать оптимизацию бизнес-метрики?

Будем исходить из предположения, что чем релевантнее объявления, тем выше вероятность кликнуть на значок "Написать" или "Показать телефон". Дополнительно предположим, что клиент вряд ли склонен листать много страниц с предложениями, так что для прототипа можно сделать выдачу только одной страницы с k объявлениями. В качестве метрики Mean average precision at k выглядит приемлемым (учитывает ранжирование и в целом попадание или непопадание нами).

Насчёт k, за отправную точку возьму число в 50 объявлений: отталкивался от кол-ва объявлений на сайте Авито, на одной странице насчитал около 70 объявлений, но решил взять чуть меньше. В целом, с выбором k есть две стороны медали. С одной стороны, в поиске мы можем выдавать хоть все возможнные объявления, главное чтобы в начале были более релевантные запросу. Но, с другой стороны, выдавать всё нельзя, ибо это очень странно если при условном поиске смартфонов на какой-нибудь странице мы начнём выдавать недвижимость. Нужно будет прийти к решению, где k будет подбираться под запрос (под запрос должны подходить только те объявления, которые прошли определённый порог метрики релевантности, подбирать по метрикам, например, F1-score).
