import logging

import azure.functions as func

""" Adapted from the second, shorter solution at http://www.codecodex.com/wiki/Calculate_digits_of_pi#Python
"""

def pi_digits_Python(digits):
    scale = 10000
    maxarr = int((digits / 4) * 14)
    arrinit = 2000
    carry = 0
    arr = [arrinit] * (maxarr + 1)
    output = ""

    for i in range(maxarr, 1, -14):
        total = 0
        for j in range(i, 0, -1):
            total = (total * j) + (scale * arr[j])
            arr[j] = total % ((j * 2) - 1)
            total = total / ((j * 2) - 1)

        output += "%04d" % (carry + (total / scale))
        carry = total % scale

    return output

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('DigitsOfPi HTTP trigger function processed a request.')

    digits_param = req.params.get('digits')

    if digits_param is not None:
        try:
            digits = int(digits_param)
        except ValueError:
            digits = 10   # A default

        if digits > 0:
            digit_string = pi_digits_Python(digits)

            # Insert a decimal point in the return value
            return func.HttpResponse(digit_string[:1] + '.' + digit_string[1:])

    return func.HttpResponse(
         "Please pass the URL parameter ?digits= to specify a positive number of digits.",
         status_code=400
    )