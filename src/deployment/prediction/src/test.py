import unittest
from predict import lambda_handler


class TestLambdaHandlerLocal(unittest.TestCase):
    """Test the lambda handler functionality on the local machine."""

    def test_outlier(self):
        """Ensure returns outlier when values for hr are high."""
        event_outl = {"body": "1,92.5692246835,16.613924050599998,1,2000-03-20 19:40:00"}
        result_outl = lambda_handler(event_outl, {})
        self.assertEqual(int(result_outl['body']), -1)

    def test_normal(self):
        """Ensure returns normal output when values for hr are in the normal range."""
        event_norm = {"body": "1,62.5692246835,16.613924050599998,1,2000-03-20 19:40:00"}
        result_norm = lambda_handler(event_norm, {})
        self.assertEqual(int(result_norm['body']), 1)

    def test_malformed_column_input(self):
        """Ensure returns malformed input when number of columns is wrong."""
        event_norm = {"body": "1,62.5692246835,16.613924050599998,1"}
        result_norm = lambda_handler(event_norm, {})
        self.assertEqual(result_norm['body'], '"Malformed data input."')


if __name__ == "__main__":
    unittest.main()
