BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "cars" (
	"Model"	TEXT,
	"MPG"	REAL,
	"Cylinders"	INTEGER,
	"Displacement"	INTEGER,
	"Horsepower"	INTEGER,
	"Weight"	INTEGER,
	"Acceleration"	REAL,
	"Year"	INTEGER,
	"Origin"	TEXT
);
INSERT INTO "cars" VALUES ('volkswagen 1131 deluxe sedan',26.0,4,97,46,1835,20.5,1970,'Europe');
INSERT INTO "cars" VALUES ('volkswagen super beetle',26.0,4,97,46,1950,21.0,1973,'Europe');
INSERT INTO "cars" VALUES ('volkswagen rabbit custom diesel',43.1,4,90,48,1985,21.5,1978,'Europe');
INSERT INTO "cars" VALUES ('vw rabbit c (diesel)',44.3,4,90,48,2085,21.7,1980,'Europe');
INSERT INTO "cars" VALUES ('vw dasher (diesel)',43.4,4,90,48,2335,23.7,1980,'Europe');
INSERT INTO "cars" VALUES ('fiat 128',29.0,4,68,49,1867,19.5,1973,'Europe');
INSERT INTO "cars" VALUES ('toyota corona',31.0,4,76,52,1649,16.5,1974,'Japan');
INSERT INTO "cars" VALUES ('chevrolet chevette',29.0,4,85,52,2035,22.2,1976,'US');
INSERT INTO "cars" VALUES ('mazda glc deluxe',32.8,4,78,52,1985,19.4,1978,'Japan');
INSERT INTO "cars" VALUES ('vw pickup',44.0,4,97,52,2130,24.6,1982,'Europe');
INSERT INTO "cars" VALUES ('honda civic cvcc',33.0,4,91,53,1795,17.5,1975,'Japan');
INSERT INTO "cars" VALUES ('honda civic',33.0,4,91,53,1795,17.4,1976,'Japan');
INSERT INTO "cars" VALUES ('volkswagen type 3',23.0,4,97,54,2254,23.5,1972,'Europe');
INSERT INTO "cars" VALUES ('renault 5 gtl',36.0,4,79,58,1825,18.6,1977,'Europe');
INSERT INTO "cars" VALUES ('toyota starlet',39.1,4,79,58,1755,16.9,1981,'Japan');
INSERT INTO "cars" VALUES ('volkswagen model 111',27.0,4,97,60,1834,19.0,1971,'Europe');
INSERT INTO "cars" VALUES ('chevrolet woody',24.5,4,98,60,2164,22.1,1976,'US');
INSERT INTO "cars" VALUES ('toyota corolla tercel',38.1,4,89,60,1968,18.8,1980,'Japan');
INSERT INTO "cars" VALUES ('honda civic 1300',35.1,4,81,60,1760,16.1,1981,'Japan');
INSERT INTO "cars" VALUES ('datsun 710',32.0,4,83,61,2003,19.0,1974,'Japan');
INSERT INTO "cars" VALUES ('vokswagen rabbit',29.8,4,89,62,1845,15.3,1980,'Europe');
INSERT INTO "cars" VALUES ('toyota tercel',37.7,4,89,62,2050,17.3,1981,'Japan');
INSERT INTO "cars" VALUES ('plymouth horizon 4',34.7,4,105,63,2215,14.9,1981,'US');
INSERT INTO "cars" VALUES ('plymouth horizon miser',38.0,4,105,63,2125,14.7,1982,'US');
INSERT INTO "cars" VALUES ('plymouth champ',39.0,4,86,64,1875,16.4,1981,'US');
INSERT INTO "cars" VALUES ('toyota corolla 1200',31.0,4,71,65,1773,19.0,1971,'Japan');
INSERT INTO "cars" VALUES ('maxda glc deluxe',34.1,4,86,65,1975,15.2,1979,'Japan');
INSERT INTO "cars" VALUES ('datsun 210',31.8,4,85,65,2020,19.2,1979,'Japan');
INSERT INTO "cars" VALUES ('datsun 310',37.2,4,86,65,2019,16.4,1980,'Japan');
INSERT INTO "cars" VALUES ('mazda glc',46.6,4,86,65,2110,17.9,1980,'Japan');
INSERT INTO "cars" VALUES ('datsun 210 mpg',37.0,4,85,65,1975,19.4,1981,'Japan');
INSERT INTO "cars" VALUES ('ford escort 4w',34.4,4,98,65,2045,16.2,1981,'US');
INSERT INTO "cars" VALUES ('ford escort 2h',29.9,4,98,65,2380,20.7,1981,'US');
INSERT INTO "cars" VALUES ('ford fiesta',36.1,4,98,66,1800,14.4,1978,'US');
INSERT INTO "cars" VALUES ('datsun b210',31.0,4,79,67,1950,19.0,1974,'Japan');
INSERT INTO "cars" VALUES ('volkswagen dasher',26.0,4,79,67,1963,15.5,1974,'Europe');
INSERT INTO "cars" VALUES ('fiat x1.9',31.0,4,79,67,2000,16.0,1974,'Europe');
INSERT INTO "cars" VALUES ('subaru dl',30.0,4,97,67,1985,16.4,1977,'Japan');
INSERT INTO "cars" VALUES ('audi 5000s (diesel)',36.4,5,121,67,2950,19.9,1980,'Europe');
INSERT INTO "cars" VALUES ('mercedes-benz 240d',30.0,4,146,67,3250,21.8,1980,'Europe');
INSERT INTO "cars" VALUES ('honda civic 1500 gl',44.6,4,91,67,1850,13.8,1980,'Japan');
INSERT INTO "cars" VALUES ('subaru',32.3,4,97,67,2065,17.8,1981,'Japan');
INSERT INTO "cars" VALUES ('honda civic (auto)',32.0,4,91,67,1965,15.7,1982,'Japan');
INSERT INTO "cars" VALUES ('datsun 310 gx',38.0,4,91,67,1995,16.2,1982,'Japan');
INSERT INTO "cars" VALUES ('honda accord cvcc',31.5,4,98,68,2045,18.5,1977,'Japan');
INSERT INTO "cars" VALUES ('honda accord lx',29.5,4,98,68,2135,16.6,1978,'Japan');
INSERT INTO "cars" VALUES ('mazda glc 4',34.1,4,91,68,1985,16.0,1981,'Japan');
INSERT INTO "cars" VALUES ('mazda glc custom l',37.0,4,91,68,2025,18.2,1982,'Japan');
INSERT INTO "cars" VALUES ('mazda glc custom',31.0,4,91,68,1970,17.6,1982,'Japan');
INSERT INTO "cars" VALUES ('datsun 1200',35.0,4,72,69,1613,18.0,1971,'Japan');
INSERT INTO "cars" VALUES ('renault 12 (sw)',26.0,4,96,69,2189,18.0,1972,'Europe');
INSERT INTO "cars" VALUES ('fiat strada custom',37.3,4,91,69,2130,14.7,1979,'Europe');
INSERT INTO "cars" VALUES ('peugeot 304',30.0,4,79,70,2074,19.5,1971,'Europe');
INSERT INTO "cars" VALUES ('plymouth cricket',26.0,4,91,70,1955,20.5,1971,'US');
INSERT INTO "cars" VALUES ('volkswagen rabbit',29.0,4,90,70,1937,14.0,1975,'Europe');
INSERT INTO "cars" VALUES ('vw rabbit',29.0,4,90,70,1937,14.2,1976,'Europe');
INSERT INTO "cars" VALUES ('datsun b-210',32.0,4,85,70,1990,17.0,1976,'Japan');
INSERT INTO "cars" VALUES ('datsun f-10 hatchback',33.5,4,85,70,1945,16.8,1977,'Japan');
INSERT INTO "cars" VALUES ('datsun b210 gx',39.4,4,85,70,2070,18.6,1978,'Japan');
INSERT INTO "cars" VALUES ('plymouth horizon',34.2,4,105,70,2200,13.2,1979,'US');
INSERT INTO "cars" VALUES ('plymouth horizon tc3',34.5,4,105,70,2150,14.9,1979,'US');
INSERT INTO "cars" VALUES ('mercury lynx l',36.0,4,98,70,2125,17.3,1982,'US');
INSERT INTO "cars" VALUES ('toyota corolla',34.0,4,108,70,2245,16.9,1982,'Japan');
INSERT INTO "cars" VALUES ('volkswagen scirocco',31.5,4,89,71,1990,14.9,1978,'Europe');
INSERT INTO "cars" VALUES ('vw rabbit custom',31.9,4,89,71,1925,14.0,1979,'Europe');
INSERT INTO "cars" VALUES ('peugeot 504',27.2,4,141,71,3190,24.8,1979,'Europe');
INSERT INTO "cars" VALUES ('chevrolet vega (sw)',22.0,4,140,72,2408,19.0,1971,'US');
INSERT INTO "cars" VALUES ('chevrolet vega',21.0,4,140,72,2401,19.5,1973,'US');
INSERT INTO "cars" VALUES ('mercury monarch',15.0,6,250,72,3432,21.0,1975,'US');
INSERT INTO "cars" VALUES ('ford maverick',15.0,6,250,72,3158,19.5,1975,'US');
INSERT INTO "cars" VALUES ('ford pinto',26.5,4,140,72,2565,13.6,1976,'US');
INSERT INTO "cars" VALUES ('honda accord',32.4,4,107,72,2290,17.0,1980,'Japan');
INSERT INTO "cars" VALUES ('volkswagen jetta',33.0,4,105,74,2190,14.2,1981,'Europe');
INSERT INTO "cars" VALUES ('mazda 626',31.6,4,120,74,2635,18.3,1981,'Japan');
INSERT INTO "cars" VALUES ('volkswagen rabbit l',36.0,4,105,74,1980,15.3,1982,'Europe');
INSERT INTO "cars" VALUES ('opel manta',24.0,4,116,75,2158,15.5,1973,'Europe');
INSERT INTO "cars" VALUES ('dodge colt',28.0,4,90,75,2125,14.5,1974,'US');
INSERT INTO "cars" VALUES ('fiat 124 tc',26.0,4,116,75,2246,14.0,1974,'Europe');
INSERT INTO "cars" VALUES ('toyota corolla liftback',26.0,4,97,75,2265,18.2,1977,'Japan');
INSERT INTO "cars" VALUES ('dodge omni',30.9,4,105,75,2230,14.5,1978,'US');
INSERT INTO "cars" VALUES ('honda prelude',33.7,4,107,75,2210,14.4,1981,'Japan');
INSERT INTO "cars" VALUES ('fiat 124b',30.0,4,88,76,2065,14.5,1971,'Europe');
INSERT INTO "cars" VALUES ('volkswagen 411 (sw)',22.0,4,121,76,2511,18.0,1972,'Europe');
INSERT INTO "cars" VALUES ('volvo diesel',30.7,6,145,76,3160,19.6,1981,'Europe');
INSERT INTO "cars" VALUES ('mercedes benz 300d',25.4,5,183,77,3530,20.1,1979,'Europe');
INSERT INTO "cars" VALUES ('pontiac astro',23.0,4,140,78,2592,18.5,1975,'US');
INSERT INTO "cars" VALUES ('ford granada ghia',18.0,6,250,78,3574,21.0,1976,'US');
INSERT INTO "cars" VALUES ('volkswagen rabbit custom',29.0,4,97,78,1940,14.5,1977,'Europe');
INSERT INTO "cars" VALUES ('audi 4000',34.3,4,97,78,2188,15.8,1980,'Europe');
INSERT INTO "cars" VALUES ('ford ranger',28.0,4,120,79,2625,18.6,1982,'US');
INSERT INTO "cars" VALUES ('dodge colt hardtop',25.0,4,97.5,80,2126,17.0,1972,'US');
INSERT INTO "cars" VALUES ('dodge colt (sw)',28.0,4,98,80,2164,15.0,1972,'US');
INSERT INTO "cars" VALUES ('buick opel isuzu deluxe',30.0,4,111,80,2155,14.8,1977,'US');
INSERT INTO "cars" VALUES ('dodge colt hatchback custom',35.7,4,98,80,1915,14.4,1979,'US');
INSERT INTO "cars" VALUES ('amc spirit dl',27.4,4,121,80,2670,15.0,1979,'US');
INSERT INTO "cars" VALUES ('peugeot 505s turbo diesel',28.1,4,141,80,3230,20.4,1981,'Europe');
INSERT INTO "cars" VALUES ('opel 1900',25.0,4,116,81,2220,16.9,1976,'Europe');
INSERT INTO "cars" VALUES ('chevy s-10',31.0,4,119,82,2720,19.4,1982,'US');
INSERT INTO "cars" VALUES ('audi fox',29.0,4,98,83,2219,16.5,1974,'Europe');
INSERT INTO "cars" VALUES ('renault 12tl',27.0,4,101,83,2202,15.3,1976,'Europe');
INSERT INTO "cars" VALUES ('dodge colt m/m',33.5,4,98,83,2075,15.9,1977,'US');
INSERT INTO "cars" VALUES ('plymouth reliant',27.2,4,135,84,2490,15.7,1981,'US');
INSERT INTO "cars" VALUES ('buick skylark',26.6,4,151,84,2635,16.4,1981,'US');
INSERT INTO "cars" VALUES ('dodge aries se',29.0,4,135,84,2525,16.0,1982,'US');
INSERT INTO "cars" VALUES ('dodge charger 2.2',36.0,4,135,84,2370,13.0,1982,'US');
INSERT INTO "cars" VALUES ('dodge rampage',32.0,4,135,84,2295,11.6,1982,'US');
INSERT INTO "cars" VALUES ('ford fairmont (auto)',20.2,6,200,85,2965,15.8,1978,'US');
INSERT INTO "cars" VALUES ('mercury zephyr',20.8,6,200,85,3070,16.7,1978,'US');
INSERT INTO "cars" VALUES ('oldsmobile starfire sx',23.8,4,151,85,2855,17.6,1978,'US');
INSERT INTO "cars" VALUES ('mercury zephyr 6',19.8,6,200,85,2990,18.2,1979,'US');
INSERT INTO "cars" VALUES ('chrysler lebaron salon',17.6,6,225,85,3465,16.6,1981,'US');
INSERT INTO "cars" VALUES ('pontiac j2000 se hatchback',31.0,4,112,85,2575,16.2,1982,'US');
INSERT INTO "cars" VALUES ('oldsmobile cutlass ciera (diesel)',38.0,6,262,85,3015,17.0,1982,'US');
INSERT INTO "cars" VALUES ('mercury capri 2000',23.0,4,122,86,2220,14.0,1971,'US');
INSERT INTO "cars" VALUES ('ford pinto runabout',21.0,4,122,86,2226,16.5,1972,'US');
INSERT INTO "cars" VALUES ('ford pinto (sw)',22.0,4,122,86,2395,16.0,1972,'US');
INSERT INTO "cars" VALUES ('fiat 131',28.0,4,107,86,2464,15.5,1976,'Europe');
INSERT INTO "cars" VALUES ('ford mustang gl',27.0,4,140,86,2790,15.6,1982,'US');
INSERT INTO "cars" VALUES ('peugeot 504 (sw)',21.0,4,120,87,2979,19.5,1972,'Europe');
INSERT INTO "cars" VALUES ('datsun pl510',27.0,4,97,88,2130,14.5,1970,'Japan');
INSERT INTO "cars" VALUES ('ford torino 500',19.0,6,250,88,3302,15.5,1971,'US');
INSERT INTO "cars" VALUES ('ford mustang',18.0,6,250,88,3139,14.5,1971,'US');
INSERT INTO "cars" VALUES ('toyota corolla 1600 (sw)',27.0,4,97,88,2100,16.5,1972,'Japan');
INSERT INTO "cars" VALUES ('toyota carina',20.0,4,97,88,2279,19.0,1973,'Japan');
INSERT INTO "cars" VALUES ('pontiac sunbird coupe',24.5,4,151,88,2740,16.0,1977,'US');
INSERT INTO "cars" VALUES ('ford fairmont (man)',25.1,4,140,88,2720,15.4,1978,'US');
INSERT INTO "cars" VALUES ('ford fairmont 4',22.3,4,140,88,2890,17.3,1979,'US');
INSERT INTO "cars" VALUES ('ford fairmont',26.4,4,140,88,2870,18.1,1980,'US');
INSERT INTO "cars" VALUES ('triumph tr7 coupe',35.0,4,122,88,2500,15.1,1980,'Europe');
INSERT INTO "cars" VALUES ('ford granada gl',20.2,6,200,88,3060,17.1,1981,'US');
INSERT INTO "cars" VALUES ('chevrolet cavalier',28.0,4,112,88,2605,19.6,1982,'US');
INSERT INTO "cars" VALUES ('chevrolet cavalier wagon',27.0,4,112,88,2640,18.6,1982,'US');
INSERT INTO "cars" VALUES ('chevrolet cavalier 2-door',34.0,4,112,88,2395,18.0,1982,'US');
INSERT INTO "cars" VALUES ('nissan stanza xe',36.0,4,120,88,2160,14.5,1982,'Japan');
INSERT INTO "cars" VALUES ('ford mustang ii 2+2',25.5,4,140,89,2755,15.8,1977,'US');
INSERT INTO "cars" VALUES ('audi 100 ls',24.0,4,107,90,2430,14.5,1970,'Europe');
INSERT INTO "cars" VALUES ('amc gremlin',21.0,6,199,90,2648,15.0,1970,'US');
INSERT INTO "cars" VALUES ('chevrolet vega 2300',28.0,4,140,90,2264,15.5,1971,'US');
INSERT INTO "cars" VALUES ('maxda rx3',18.0,3,70,90,2124,13.5,1973,'Japan');
INSERT INTO "cars" VALUES ('fiat 124 sport coupe',26.0,4,98,90,2265,15.5,1973,'Europe');
INSERT INTO "cars" VALUES ('amc pacer',19.0,6,232,90,3211,17.0,1975,'US');
INSERT INTO "cars" VALUES ('amc hornet',22.5,6,232,90,3085,17.6,1976,'US');
INSERT INTO "cars" VALUES ('amc concord',19.4,6,232,90,3210,17.2,1978,'US');
INSERT INTO "cars" VALUES ('amc concord dl 6',20.2,6,232,90,3265,18.2,1979,'US');
INSERT INTO "cars" VALUES ('oldsmobile cutlass salon brougham',23.9,8,260,90,3420,22.2,1979,'US');
INSERT INTO "cars" VALUES ('buick skylark limited',28.4,4,151,90,2670,16.0,1979,'US');
INSERT INTO "cars" VALUES ('pontiac phoenix',33.5,4,151,90,2556,13.2,1979,'US');
INSERT INTO "cars" VALUES ('chevrolet citation',28.0,4,151,90,2678,16.5,1980,'US');
INSERT INTO "cars" VALUES ('dodge aspen',19.1,6,225,90,3381,18.7,1980,'US');
INSERT INTO "cars" VALUES ('toyota corona liftback',29.8,4,134,90,2711,15.5,1980,'Japan');
INSERT INTO "cars" VALUES ('chevrolet camaro',27.0,4,151,90,2950,17.3,1982,'US');
INSERT INTO "cars" VALUES ('audi 100ls',20.0,4,114,91,2582,14.0,1973,'Europe');
INSERT INTO "cars" VALUES ('datsun 510 (sw)',28.0,4,97,92,2288,17.0,1972,'Japan');
INSERT INTO "cars" VALUES ('capri ii',25.0,4,140,92,2572,14.9,1976,'US');
INSERT INTO "cars" VALUES ('datsun 510 hatchback',37.0,4,119,92,2434,15.0,1980,'Japan');
INSERT INTO "cars" VALUES ('dodge aries wagon (sw)',25.8,4,156,92,2620,14.4,1981,'US');
INSERT INTO "cars" VALUES ('ford fairmont futura',24.0,4,140,92,2865,16.4,1982,'US');
INSERT INTO "cars" VALUES ('chrysler lebaron medallion',26.0,4,156,92,2585,14.5,1982,'US');
INSERT INTO "cars" VALUES ('datsun 610',22.0,4,108,94,2379,16.5,1973,'Japan');
INSERT INTO "cars" VALUES ('toyota corona mark ii',24.0,4,113,95,2372,15.0,1970,'Japan');
INSERT INTO "cars" VALUES ('plymouth duster',22.0,6,198,95,2833,15.5,1970,'US');
INSERT INTO "cars" VALUES ('saab 99e',25.0,4,104,95,2375,17.5,1970,'Europe');
INSERT INTO "cars" VALUES ('toyota corona hardtop',24.0,4,113,95,2278,15.5,1972,'Japan');
INSERT INTO "cars" VALUES ('plymouth valiant custom',19.0,6,225,95,3264,16.0,1975,'US');
INSERT INTO "cars" VALUES ('plymouth fury',18.0,6,225,95,3785,19.0,1975,'US');
INSERT INTO "cars" VALUES ('amc pacer d/l',17.5,6,258,95,3193,17.8,1976,'US');
INSERT INTO "cars" VALUES ('chevrolet malibu',20.5,6,200,95,3155,18.2,1978,'US');
INSERT INTO "cars" VALUES ('toyota celica gt liftback',21.1,4,134,95,2515,14.8,1978,'Japan');
INSERT INTO "cars" VALUES ('plymouth arrow gs',25.5,4,122,96,2300,15.5,1977,'US');
INSERT INTO "cars" VALUES ('toyota celica gt',32.0,4,144,96,2665,13.9,1982,'Japan');
INSERT INTO "cars" VALUES ('mazda rx2 coupe',19.0,3,70,97,2330,13.5,1972,'Japan');
INSERT INTO "cars" VALUES ('toyouta corona mark ii (sw)',23.0,4,120,97,2506,14.5,1972,'Japan');
INSERT INTO "cars" VALUES ('datsun 810',22.0,6,146,97,2815,14.5,1977,'Japan');
INSERT INTO "cars" VALUES ('datsun 510',27.2,4,119,97,2300,14.7,1978,'Japan');
INSERT INTO "cars" VALUES ('datsun 200-sx',23.9,4,119,97,2405,14.9,1978,'Japan');
INSERT INTO "cars" VALUES ('volvo 244dl',22.0,4,121,98,2945,14.5,1975,'Europe');
INSERT INTO "cars" VALUES ('ford granada',18.5,6,250,98,3525,19.0,1977,'US');
INSERT INTO "cars" VALUES ('chevrolet chevelle malibu',17.0,6,250,100,3329,15.5,1971,'US');
INSERT INTO "cars" VALUES ('amc matador',18.0,6,232,100,3288,15.5,1971,'US');
INSERT INTO "cars" VALUES ('pontiac firebird',19.0,6,250,100,3282,15.0,1971,'US');
INSERT INTO "cars" VALUES ('chevrolet nova custom',16.0,6,250,100,3278,18.0,1973,'US');
INSERT INTO "cars" VALUES ('chevrolet nova',15.0,6,250,100,3336,17.0,1974,'US');
INSERT INTO "cars" VALUES ('chevrolet chevelle malibu classic',16.0,6,250,100,3781,17.0,1974,'US');
INSERT INTO "cars" VALUES ('plymouth valiant',22.0,6,225,100,3233,15.4,1976,'US');
INSERT INTO "cars" VALUES ('dodge aspen se',20.0,6,225,100,3651,17.7,1976,'US');
INSERT INTO "cars" VALUES ('plymouth volare custom',19.0,6,225,100,3630,17.7,1977,'US');
INSERT INTO "cars" VALUES ('plymouth volare',20.5,6,225,100,3430,17.2,1978,'US');
INSERT INTO "cars" VALUES ('mazda rx-7 gs',23.7,3,70,100,2420,12.5,1980,'Japan');
INSERT INTO "cars" VALUES ('datsun 200sx',32.9,4,119,100,2615,14.8,1981,'Japan');
INSERT INTO "cars" VALUES ('volvo 245',20.0,4,130,102,3150,15.7,1976,'Europe');
INSERT INTO "cars" VALUES ('audi 5000',20.3,5,131,103,2830,15.9,1978,'Europe');
INSERT INTO "cars" VALUES ('plymouth satellite custom',16.0,6,225,105,3439,15.5,1971,'US');
INSERT INTO "cars" VALUES ('plymouth satellite sebring',18.0,6,225,105,3613,16.5,1974,'US');
INSERT INTO "cars" VALUES ('chevroelt chevelle malibu',16.0,6,250,105,3897,18.5,1975,'US');
INSERT INTO "cars" VALUES ('pontiac phoenix lj',19.2,6,231,105,3535,19.2,1978,'US');
INSERT INTO "cars" VALUES ('buick century special',20.6,6,231,105,3380,15.8,1978,'US');
INSERT INTO "cars" VALUES ('plymouth sapporo',23.2,4,156,105,2745,16.7,1978,'US');
INSERT INTO "cars" VALUES ('oldsmobile cutlass ls',26.6,8,350,105,3725,19.0,1981,'US');
INSERT INTO "cars" VALUES ('mercury capri v6',21.0,6,155,107,2472,14.0,1973,'US');
INSERT INTO "cars" VALUES ('toyota mark ii',19.0,6,156,108,2930,15.5,1976,'Japan');
INSERT INTO "cars" VALUES ('amc hornet sportabout (sw)',18.0,6,258,110,2962,13.5,1971,'US');
INSERT INTO "cars" VALUES ('saab 99le',24.0,4,121,110,2660,14.0,1973,'Europe');
INSERT INTO "cars" VALUES ('buick century',17.0,6,231,110,3907,21.0,1975,'US');
INSERT INTO "cars" VALUES ('buick skyhawk',21.0,6,231,110,3039,15.0,1975,'US');
INSERT INTO "cars" VALUES ('chevrolet monza 2+2',20.0,8,262,110,3221,13.5,1975,'US');
INSERT INTO "cars" VALUES ('pontiac ventura sj',18.5,6,250,110,3645,16.2,1976,'US');
INSERT INTO "cars" VALUES ('oldsmobile cutlass supreme',17.0,8,260,110,4060,19.0,1977,'US');
INSERT INTO "cars" VALUES ('chevrolet concours',17.5,6,250,110,3520,16.4,1977,'US');
INSERT INTO "cars" VALUES ('bmw 320i',21.5,4,121,110,2600,12.8,1977,'Europe');
INSERT INTO "cars" VALUES ('mazda rx-4',21.5,3,80,110,2720,13.5,1977,'Japan');
INSERT INTO "cars" VALUES ('dodge aspen 6',20.6,6,225,110,3360,16.6,1979,'US');
INSERT INTO "cars" VALUES ('buick century limited',25.0,6,181,110,2945,16.4,1982,'US');
INSERT INTO "cars" VALUES ('volvo 145e (sw)',18.0,4,121,112,2933,14.5,1972,'Europe');
INSERT INTO "cars" VALUES ('volvo 144ea',19.0,4,121,112,2868,15.5,1973,'Europe');
INSERT INTO "cars" VALUES ('ford granada l',22.0,6,232,112,2835,14.7,1982,'US');
INSERT INTO "cars" VALUES ('bmw 2002',26.0,4,121,113,2234,12.5,1970,'Europe');
INSERT INTO "cars" VALUES ('saab 99gle',21.6,4,121,115,2795,15.7,1978,'Europe');
INSERT INTO "cars" VALUES ('pontiac lemans v6',21.5,6,231,115,3245,15.4,1979,'US');
INSERT INTO "cars" VALUES ('oldsmobile omega brougham',26.8,6,173,115,2700,12.9,1979,'US');
INSERT INTO "cars" VALUES ('toyota cressida',25.4,6,168,116,2900,12.6,1981,'Japan');
INSERT INTO "cars" VALUES ('mercedes-benz 280s',16.5,6,168,120,3820,16.7,1976,'Europe');
INSERT INTO "cars" VALUES ('amc concord dl 2',18.1,6,258,120,3410,15.1,1978,'US');
INSERT INTO "cars" VALUES ('datsun 810 maxima',24.2,6,146,120,2930,13.8,1981,'Japan');
INSERT INTO "cars" VALUES ('volvo 264gl',17.0,6,163,125,3140,13.6,1978,'Europe');
INSERT INTO "cars" VALUES ('chevrolet malibu classic (sw)',19.2,8,267,125,3605,15.0,1979,'US');
INSERT INTO "cars" VALUES ('cadillac eldorado',23.0,8,350,125,3900,17.4,1979,'US');
INSERT INTO "cars" VALUES ('ford mustang ii',13.0,8,302,129,3169,12.0,1975,'US');
INSERT INTO "cars" VALUES ('ford ltd landau',17.6,8,302,129,3725,13.4,1979,'US');
INSERT INTO "cars" VALUES ('chevrolet chevelle concours (sw)',13.0,8,307,130,4098,14.0,1972,'US');
INSERT INTO "cars" VALUES ('ford f108',13.0,8,302,130,3870,15.0,1976,'US');
INSERT INTO "cars" VALUES ('mercury cougar brougham',15.0,8,302,130,4295,14.9,1977,'US');
INSERT INTO "cars" VALUES ('chevrolet caprice classic',17.0,8,305,130,3840,15.4,1979,'US');
INSERT INTO "cars" VALUES ('datsun 280-zx',32.7,6,168,132,2910,11.4,1980,'Japan');
INSERT INTO "cars" VALUES ('peugeot 604sl',16.2,6,163,133,3410,15.8,1978,'Europe');
INSERT INTO "cars" VALUES ('dodge st. regis',18.2,8,318,135,3830,15.2,1979,'US');
INSERT INTO "cars" VALUES ('ford gran torino',14.0,8,302,137,4042,14.5,1973,'US');
INSERT INTO "cars" VALUES ('mercury grand marquis',16.5,8,351,138,3955,13.2,1979,'US');
INSERT INTO "cars" VALUES ('mercury monarch ghia',20.2,8,302,139,3570,12.8,1978,'US');
INSERT INTO "cars" VALUES ('ford futura',18.1,8,302,139,3205,11.2,1978,'US');
INSERT INTO "cars" VALUES ('ford torino',17.0,8,302,140,3449,10.5,1970,'US');
INSERT INTO "cars" VALUES ('ford gran torino (sw)',13.0,8,302,140,4294,16.0,1972,'US');
INSERT INTO "cars" VALUES ('dodge diplomat',19.4,8,318,140,3735,13.2,1978,'US');
INSERT INTO "cars" VALUES ('dodge magnum xe',17.5,8,318,140,4080,13.7,1978,'US');
INSERT INTO "cars" VALUES ('ford country squire (sw)',15.5,8,351,142,4054,14.3,1979,'US');
INSERT INTO "cars" VALUES ('chevrolet monte carlo s',15.0,8,350,145,4082,13.0,1973,'US');
INSERT INTO "cars" VALUES ('chevrolet bel air',15.0,8,350,145,4440,14.0,1975,'US');
INSERT INTO "cars" VALUES ('chevy c10',13.0,8,350,145,4055,12.0,1976,'US');
INSERT INTO "cars" VALUES ('dodge monaco brougham',15.5,8,318,145,4140,13.7,1977,'US');
INSERT INTO "cars" VALUES ('chevrolet monte carlo landau',19.2,8,305,145,3425,13.2,1978,'US');
INSERT INTO "cars" VALUES ('ford ltd',14.0,8,351,148,4657,13.5,1975,'US');
INSERT INTO "cars" VALUES ('ford thunderbird',16.0,8,351,149,4335,14.5,1977,'US');
INSERT INTO "cars" VALUES ('plymouth satellite',18.0,8,318,150,3436,11.0,1970,'US');
INSERT INTO "cars" VALUES ('amc rebel sst',16.0,8,304,150,3433,12.0,1970,'US');
INSERT INTO "cars" VALUES ('chevrolet monte carlo',15.0,8,400,150,3761,9.5,1970,'US');
INSERT INTO "cars" VALUES ('plymouth fury iii',14.0,8,318,150,4096,13.0,1971,'US');
INSERT INTO "cars" VALUES ('amc ambassador sst',17.0,8,304,150,3672,11.5,1972,'US');
INSERT INTO "cars" VALUES ('amc matador (sw)',15.0,8,304,150,3892,12.5,1972,'US');
INSERT INTO "cars" VALUES ('plymouth satellite custom (sw)',14.0,8,318,150,4077,14.0,1972,'US');
INSERT INTO "cars" VALUES ('dodge coronet custom',15.0,8,318,150,3777,12.5,1973,'US');
INSERT INTO "cars" VALUES ('plymouth fury gran sedan',14.0,8,318,150,4237,14.5,1973,'US');
INSERT INTO "cars" VALUES ('chevrolet impala',11.0,8,400,150,4997,14.0,1973,'US');
INSERT INTO "cars" VALUES ('dodge dart custom',15.0,8,318,150,3399,11.0,1973,'US');
INSERT INTO "cars" VALUES ('buick century luxus (sw)',13.0,8,350,150,4699,14.5,1974,'US');
INSERT INTO "cars" VALUES ('dodge coronet custom (sw)',14.0,8,318,150,4457,13.5,1974,'US');
INSERT INTO "cars" VALUES ('plymouth grand fury',16.0,8,318,150,4498,14.5,1975,'US');
INSERT INTO "cars" VALUES ('dodge coronet brougham',16.0,8,318,150,4190,13.0,1976,'US');
INSERT INTO "cars" VALUES ('plymouth volare premier v8',13.0,8,318,150,3940,13.2,1976,'US');
INSERT INTO "cars" VALUES ('dodge d100',13.0,8,318,150,3755,14.0,1976,'US');
INSERT INTO "cars" VALUES ('chrysler lebaron town @ country (sw)',18.5,8,360,150,3940,13.0,1979,'US');
INSERT INTO "cars" VALUES ('ford galaxie 500',14.0,8,351,153,4154,13.5,1971,'US');
INSERT INTO "cars" VALUES ('buick lesabre custom',13.0,8,350,155,4502,13.5,1972,'US');
INSERT INTO "cars" VALUES ('buick estate wagon (sw)',16.9,8,350,155,4360,14.9,1979,'US');
INSERT INTO "cars" VALUES ('plymouth cuda 340',14.0,8,340,160,3609,8.0,1970,'US');
INSERT INTO "cars" VALUES ('oldsmobile delta 88 royale',12.0,8,350,160,4456,13.5,1972,'US');
INSERT INTO "cars" VALUES ('buick skylark 320',15.0,8,350,165,3693,11.5,1970,'US');
INSERT INTO "cars" VALUES ('buick regal sport coupe (turbo)',17.7,6,231,165,3445,13.4,1978,'US');
INSERT INTO "cars" VALUES ('ford country',12.0,8,400,167,4906,12.5,1973,'US');
INSERT INTO "cars" VALUES ('dodge challenger se',15.0,8,383,170,3563,10.0,1970,'US');
INSERT INTO "cars" VALUES ('plymouth custom suburb',13.0,8,360,170,4654,13.0,1973,'US');
INSERT INTO "cars" VALUES ('pontiac catalina',16.0,8,400,170,4668,11.5,1975,'US');
INSERT INTO "cars" VALUES ('pontiac catalina brougham',14.0,8,400,175,4464,11.5,1971,'US');
INSERT INTO "cars" VALUES ('pontiac safari (sw)',13.0,8,400,175,5140,12.0,1971,'US');
INSERT INTO "cars" VALUES ('buick century 350',13.0,8,350,175,4100,13.0,1973,'US');
INSERT INTO "cars" VALUES ('amc ambassador brougham',13.0,8,360,175,3821,11.0,1973,'US');
INSERT INTO "cars" VALUES ('dodge monaco (sw)',12.0,8,383,180,4955,11.5,1971,'US');
INSERT INTO "cars" VALUES ('oldsmobile vista cruiser',12.0,8,350,180,4499,12.5,1973,'US');
INSERT INTO "cars" VALUES ('oldsmobile omega',11.0,8,350,180,3664,11.0,1973,'US');
INSERT INTO "cars" VALUES ('cadillac seville',16.5,8,350,180,4380,12.1,1976,'US');
INSERT INTO "cars" VALUES ('pontiac grand prix lj',16.0,8,400,180,4220,11.1,1977,'US');
INSERT INTO "cars" VALUES ('amc ambassador dpl',15.0,8,390,190,3850,8.5,1970,'US');
INSERT INTO "cars" VALUES ('chrysler newport royal',13.0,8,400,190,4422,12.5,1972,'US');
INSERT INTO "cars" VALUES ('chrysler cordoba',15.5,8,400,190,4325,12.2,1977,'US');
INSERT INTO "cars" VALUES ('hi 1200d',9.0,8,304,193,4732,18.5,1970,'US');
INSERT INTO "cars" VALUES ('mercury marquis brougham',12.0,8,429,198,4952,11.5,1973,'US');
INSERT INTO "cars" VALUES ('chevy c20',10.0,8,307,200,4376,15.0,1970,'US');
INSERT INTO "cars" VALUES ('mercury marquis',11.0,8,429,208,4633,11.0,1972,'US');
INSERT INTO "cars" VALUES ('dodge d200',11.0,8,318,210,4382,13.5,1970,'US');
INSERT INTO "cars" VALUES ('ford f250',10.0,8,360,215,4615,14.0,1970,'US');
INSERT INTO "cars" VALUES ('chrysler new yorker brougham',13.0,8,440,215,4735,11.0,1973,'US');
INSERT INTO "cars" VALUES ('buick electra 225 custom',12.0,8,455,225,4951,11.0,1973,'US');
INSERT INTO "cars" VALUES ('pontiac grand prix',16.0,8,400,230,4278,9.5,1973,'US');
INSERT INTO "cars" VALUES ('renault 18i',34.5,4,100,0,2320,15.8,1981,'Europe');
INSERT INTO "cars" VALUES ('amc concord dl',23.0,4,151,0,3035,20.5,1982,'US');
COMMIT;
