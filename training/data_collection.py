from lyricsgenius import Genius

genius = Genius('3n_zjq1wMVomcQtziE2rQDeLKmRTEtMhUZL1_cQukANCm31_6vzfLjcFN-QxPp5J')
artist = genius.search_artist("Burna Boy", sort="title",include_features=True)
artist.save_lyrics('lyrics2',extension='txt')

