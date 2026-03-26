import math
import unittest

import app as app_module


class CarPriceAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_module.app.config.update(TESTING=True)
        cls.client = app_module.app.test_client()

    @classmethod
    def valid_payload(cls):
        metadata = cls.client.get("/api/metadata").get_json()
        company = next(iter(metadata["companies"]))

        return {
            "company_name": company,
            "model": metadata["companies"][company][0],
            "year": 2025,
            "km_driven": 0,
            "fuel_type": metadata["fuel_types"][0],
            "transmission": metadata["transmissions"][0],
            "engine_cc": 600,
            "max_power_bhp": 30,
            "condition": "Brand New",
            "mileage_kmpl": 3,
        }

    def test_index_and_chart_are_served(self):
        index_response = self.client.get("/")
        chart_response = self.client.get(
            "/graph_visualizations/1_actual_vs_predicted.png"
        )

        self.assertEqual(index_response.status_code, 200)
        self.assertIn(b"CarValue ML", index_response.data)
        self.assertEqual(chart_response.status_code, 200)
        self.assertTrue(chart_response.data.startswith(b"\x89PNG"))

    def test_health_and_metadata_are_available(self):
        health_response = self.client.get("/api/health")
        metadata_response = self.client.get("/api/metadata")
        metadata = metadata_response.get_json()

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(
            health_response.get_json(), {"model_ready": True, "status": "ok"}
        )
        self.assertEqual(metadata_response.status_code, 200)
        self.assertTrue(metadata["success"])
        self.assertEqual(metadata["training_rows"], 6000)
        self.assertTrue(metadata["companies"])

    def test_predict_rejects_non_json_request(self):
        response = self.client.post(
            "/predict", data="not-json", content_type="text/plain"
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["errors"], ["A JSON object is required"])

    def test_predict_rejects_values_outside_app_limits(self):
        payload = self.valid_payload()
        payload["year"] = 2026

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Year cannot be greater than 2025", response.get_json()["errors"])

    def test_packaged_model_predicts_at_accepted_boundaries(self):
        response = self.client.post("/api/predict", json=self.valid_payload())
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertTrue(math.isfinite(body["predicted_price_usd"]))


if __name__ == "__main__":
    unittest.main()