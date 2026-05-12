import unittest

from places_attr_conflation.retrieval import SearchResult, rank_search_results, score_search_result, select_authoritative_result
from places_attr_conflation.small_model import TinyLinearModel, TrainingExample, build_feature_vector, train_tiny_model


class RetrievalTests(unittest.TestCase):
    def test_official_layer_scores_higher_than_aggregator_layer(self):
        official = SearchResult("https://example.com/contact", "Cafe Rio", "Contact us phone address", layer="official")
        yelp = SearchResult("https://yelp.com/biz/example", "Cafe Rio", "Reviews and photos", layer="fallback")
        self.assertGreater(score_search_result(official, query='"Cafe Rio" phone'), score_search_result(yelp, query='"Cafe Rio" phone'))

    def test_freshness_layer_receives_a_small_bonus_over_fallback(self):
        freshness = SearchResult("https://example.com/current", "Cafe Rio", "Updated contact hours", layer="freshness", recency_days=10)
        fallback = SearchResult("https://example.com/current", "Cafe Rio", "Updated contact hours", layer="fallback", recency_days=10)
        self.assertGreater(score_search_result(freshness, query='"Cafe Rio" hours'), score_search_result(fallback, query='"Cafe Rio" hours'))

    def test_search_results_rank_by_authority(self):
        results = [
            SearchResult("https://yelp.com/biz/example", "Cafe Rio", "Reviews and photos", layer="fallback", recency_days=400, zombie_score=0.8),
            SearchResult("https://example.com/contact", "Cafe Rio", "Contact us phone address", layer="official", recency_days=10),
        ]
        ranked = rank_search_results(results, query='"Cafe Rio" phone')
        self.assertEqual(ranked[0].url, "https://example.com/contact")

    def test_search_results_penalize_stale_pages(self):
        fresh = SearchResult("https://example.com/contact", "Cafe Rio", "Contact us phone address", layer="official", recency_days=10)
        stale = SearchResult("https://example.com/contact", "Cafe Rio", "Former location permanently closed", layer="official", recency_days=500, zombie_score=0.9, identity_change_score=0.6)
        self.assertGreater(score_search_result(fresh, query='"Cafe Rio" contact'), score_search_result(stale, query='"Cafe Rio" contact'))

    def test_select_authoritative_result_abstains_below_threshold(self):
        result = select_authoritative_result([SearchResult("https://yelp.com/biz/example", "Cafe Rio", "Reviews", layer="fallback", recency_days=400, zombie_score=1.0)], query='"Cafe Rio"', threshold=0.9)
        self.assertIsNone(result)

    def test_model_reranking_can_prefer_learned_authority_signals(self):
        model = TinyLinearModel(
            weights={
                "source:official_site": 2.5,
                "source:aggregator": -2.0,
                "layer:official": 1.0,
                "text:contact": 0.5,
                "text:review": -0.5,
            },
            bias=0.0,
        )
        official = SearchResult("https://example.com/contact", "Cafe Rio", "Contact us phone address", layer="official", recency_days=10)
        yelp = SearchResult("https://yelp.com/biz/example", "Cafe Rio", "Reviews and photos", layer="fallback", recency_days=10)
        self.assertGreater(score_search_result(official, query='"Cafe Rio"', model=model), score_search_result(yelp, query='"Cafe Rio"', model=model))

    def test_tiny_model_training_learns_simple_authority_signal(self):
        examples = [
            TrainingExample(build_feature_vector(SearchResult("https://example.com/contact", "Cafe Rio", "Contact us phone address", layer="official", recency_days=7)), 1),
            TrainingExample(build_feature_vector(SearchResult("https://www.google.com/maps/place/example", "Cafe Rio", "Google Maps place", layer="corroboration", recency_days=14)), 1),
            TrainingExample(build_feature_vector(SearchResult("https://yelp.com/biz/example", "Cafe Rio", "Reviews and photos", layer="fallback", recency_days=300, zombie_score=0.9)), 0),
            TrainingExample(build_feature_vector(SearchResult("https://instagram.com/cafe", "Cafe Rio", "Follow us for updates", layer="fallback", recency_days=500, zombie_score=0.8)), 0),
        ]
        model = train_tiny_model(examples, epochs=60, learning_rate=0.2, l2=0.001)
        positive = SearchResult("https://example.com/contact", "Cafe Rio", "Contact us phone address", layer="official", recency_days=7)
        negative = SearchResult("https://yelp.com/biz/example", "Cafe Rio", "Reviews and photos", layer="fallback", recency_days=300, zombie_score=0.9)
        self.assertGreater(model.score_result(positive, query='"Cafe Rio"'), model.score_result(negative, query='"Cafe Rio"'))
        self.assertGreaterEqual(model.score_result(positive, query='"Cafe Rio"'), 0.5)


if __name__ == "__main__":
    unittest.main()
